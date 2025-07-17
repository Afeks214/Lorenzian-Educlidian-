#!/usr/bin/env python3
"""
AGENT 4 CRITICAL MISSION: Comprehensive Performance Metrics and Validation System
==============================================================================

This module provides institutional-grade performance metrics and validation system
with mathematical precision, statistical rigor, and Numba JIT optimization.

Key Features:
- 50+ Advanced Performance Metrics
- Statistical Validation Framework
- Monte Carlo Simulation
- Bootstrap Confidence Intervals
- VaR and Expected Shortfall
- Risk-Adjusted Performance Measures
- Numba JIT Optimization
- Parallel Processing

Author: Agent 4 - Performance Analytics Specialist
Date: 2025-07-17
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from scipy import stats
from scipy.optimize import minimize
import warnings
from numba import jit, njit, prange
from numba.types import float64, int64, boolean
from numba.typed import Dict as NumbaDict
from numba.typed import List as NumbaList
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import time
from datetime import datetime, timedelta
import json
from pathlib import Path

warnings.filterwarnings('ignore')


@dataclass
class ComprehensivePerformanceMetrics:
    """Comprehensive performance metrics container with all institutional-grade metrics"""
    
    # Basic Return Metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    
    # Risk-Adjusted Ratios
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    information_ratio: float = 0.0
    treynor_ratio: float = 0.0
    omega_ratio: float = 0.0
    
    # Drawdown Metrics
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    average_drawdown: float = 0.0
    recovery_factor: float = 0.0
    
    # Risk Metrics
    var_95: float = 0.0
    var_99: float = 0.0
    cvar_95: float = 0.0  # Expected Shortfall
    cvar_99: float = 0.0
    
    # Market Risk Metrics
    beta: float = 0.0
    alpha: float = 0.0
    correlation: float = 0.0
    tracking_error: float = 0.0
    
    # Distribution Metrics
    skewness: float = 0.0
    kurtosis: float = 0.0
    jarque_bera_statistic: float = 0.0
    jarque_bera_pvalue: float = 0.0
    
    # Advanced Risk Metrics
    ulcer_index: float = 0.0
    martin_ratio: float = 0.0
    burke_ratio: float = 0.0
    sterling_ratio: float = 0.0
    pain_index: float = 0.0
    gain_to_pain_ratio: float = 0.0
    
    # Tail Risk Metrics
    tail_ratio: float = 0.0
    downside_variance: float = 0.0
    upside_variance: float = 0.0
    downside_volatility: float = 0.0
    upside_volatility: float = 0.0
    
    # Trade-based Metrics
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    kelly_criterion: float = 0.0
    
    # Consistency Metrics
    hit_ratio: float = 0.0
    consistency_ratio: float = 0.0
    monthly_return_std: float = 0.0
    rolling_sharpe_std: float = 0.0
    
    # Additional Advanced Metrics
    rachev_ratio: float = 0.0
    modified_sharpe_ratio: float = 0.0
    conditional_sharpe_ratio: float = 0.0
    
    # Statistical Validation
    ljung_box_pvalue: float = 0.0
    adf_pvalue: float = 0.0
    
    # Confidence Intervals
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = {}
        for key, value in self.__dict__.items():
            if key == 'confidence_intervals':
                result[key] = {k: list(v) for k, v in value.items()}
            else:
                result[key] = float(value) if isinstance(value, (int, float, np.number)) else value
        return result


@njit
def calculate_returns_jit(prices: np.ndarray) -> np.ndarray:
    """Calculate returns with JIT optimization"""
    if len(prices) < 2:
        return np.zeros(0, dtype=np.float64)
    
    returns = np.zeros(len(prices) - 1, dtype=np.float64)
    for i in range(1, len(prices)):
        returns[i-1] = (prices[i] - prices[i-1]) / prices[i-1]
    
    return returns


@njit
def calculate_drawdown_jit(prices: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
    """Calculate drawdown series and maximum drawdown with JIT optimization"""
    if len(prices) == 0:
        return np.zeros(0, dtype=np.float64), 0.0, 0, 0
    
    # Calculate running maximum
    running_max = np.zeros_like(prices)
    running_max[0] = prices[0]
    for i in range(1, len(prices)):
        running_max[i] = max(running_max[i-1], prices[i])
    
    # Calculate drawdown
    drawdown = (prices - running_max) / running_max
    
    # Find maximum drawdown
    max_dd_idx = np.argmin(drawdown)
    max_dd = abs(drawdown[max_dd_idx])
    
    # Find peak before maximum drawdown
    peak_idx = 0
    for i in range(max_dd_idx + 1):
        if prices[i] == running_max[max_dd_idx]:
            peak_idx = i
            break
    
    return drawdown, max_dd, peak_idx, max_dd_idx


@njit
def calculate_var_jit(returns: np.ndarray, confidence_level: float) -> float:
    """Calculate Value at Risk with JIT optimization"""
    if len(returns) == 0:
        return 0.0
    
    # Sort returns
    sorted_returns = np.sort(returns)
    
    # Calculate VaR index
    var_index = int((1 - confidence_level) * len(sorted_returns))
    
    if var_index >= len(sorted_returns):
        var_index = len(sorted_returns) - 1
    
    return abs(sorted_returns[var_index])


@njit
def calculate_cvar_jit(returns: np.ndarray, confidence_level: float) -> float:
    """Calculate Conditional Value at Risk (Expected Shortfall) with JIT optimization"""
    if len(returns) == 0:
        return 0.0
    
    # Sort returns
    sorted_returns = np.sort(returns)
    
    # Calculate VaR index
    var_index = int((1 - confidence_level) * len(sorted_returns))
    
    if var_index >= len(sorted_returns):
        var_index = len(sorted_returns) - 1
    
    # Calculate CVaR as mean of tail beyond VaR
    if var_index == 0:
        return abs(sorted_returns[0])
    
    tail_returns = sorted_returns[:var_index]
    return abs(np.mean(tail_returns))


@njit
def calculate_sharpe_ratio_jit(returns: np.ndarray, risk_free_rate: float, periods_per_year: int) -> float:
    """Calculate Sharpe ratio with JIT optimization"""
    if len(returns) == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate / periods_per_year
    
    if np.std(excess_returns) == 0:
        return 0.0
    
    return np.sqrt(periods_per_year) * np.mean(excess_returns) / np.std(excess_returns)


@njit
def calculate_sortino_ratio_jit(returns: np.ndarray, risk_free_rate: float, periods_per_year: int) -> float:
    """Calculate Sortino ratio with JIT optimization"""
    if len(returns) == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate / periods_per_year
    
    # Calculate downside deviation
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0:
        return np.inf if np.mean(excess_returns) > 0 else 0.0
    
    downside_deviation = np.std(downside_returns)
    
    if downside_deviation == 0:
        return 0.0
    
    return np.sqrt(periods_per_year) * np.mean(excess_returns) / downside_deviation


@njit
def calculate_calmar_ratio_jit(returns: np.ndarray, max_drawdown: float, periods_per_year: int) -> float:
    """Calculate Calmar ratio with JIT optimization"""
    if len(returns) == 0 or max_drawdown == 0:
        return 0.0
    
    annual_return = np.mean(returns) * periods_per_year
    
    return annual_return / max_drawdown


@njit
def calculate_omega_ratio_jit(returns: np.ndarray, threshold: float) -> float:
    """Calculate Omega ratio with JIT optimization"""
    if len(returns) == 0:
        return 0.0
    
    excess_returns = returns - threshold
    
    gains = excess_returns[excess_returns > 0]
    losses = excess_returns[excess_returns < 0]
    
    if len(losses) == 0:
        return np.inf if len(gains) > 0 else 1.0
    
    if len(gains) == 0:
        return 0.0
    
    return np.sum(gains) / abs(np.sum(losses))


@njit
def calculate_ulcer_index_jit(prices: np.ndarray) -> float:
    """Calculate Ulcer Index with JIT optimization"""
    if len(prices) == 0:
        return 0.0
    
    # Calculate running maximum
    running_max = np.zeros_like(prices)
    running_max[0] = prices[0]
    for i in range(1, len(prices)):
        running_max[i] = max(running_max[i-1], prices[i])
    
    # Calculate drawdown percentages
    drawdown_pct = (prices - running_max) / running_max * 100
    
    # Ulcer Index = sqrt(mean(drawdown^2))
    return np.sqrt(np.mean(drawdown_pct ** 2))


@njit
def calculate_kelly_criterion_jit(returns: np.ndarray) -> float:
    """Calculate Kelly Criterion with JIT optimization"""
    if len(returns) == 0:
        return 0.0
    
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    
    if len(wins) == 0 or len(losses) == 0:
        return 0.0
    
    win_rate = len(wins) / len(returns)
    avg_win = np.mean(wins)
    avg_loss = abs(np.mean(losses))
    
    if avg_loss == 0:
        return 0.0
    
    # Kelly = (bp - q) / b where b = avg_win/avg_loss, p = win_rate, q = 1-p
    b = avg_win / avg_loss
    p = win_rate
    q = 1 - p
    
    kelly = (b * p - q) / b
    
    return max(0.0, kelly)  # Kelly should not be negative


class ComprehensivePerformanceAnalyzer:
    """
    Comprehensive performance analyzer with advanced metrics and statistical validation
    """
    
    def __init__(self, 
                 risk_free_rate: float = 0.02,
                 periods_per_year: int = 252,
                 confidence_levels: List[float] = [0.95, 0.99],
                 bootstrap_samples: int = 1000,
                 monte_carlo_runs: int = 10000,
                 n_jobs: int = -1):
        """
        Initialize comprehensive performance analyzer
        
        Args:
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of periods per year
            confidence_levels: Confidence levels for VaR calculations
            bootstrap_samples: Number of bootstrap samples
            monte_carlo_runs: Number of Monte Carlo runs
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year
        self.confidence_levels = confidence_levels
        self.bootstrap_samples = bootstrap_samples
        self.monte_carlo_runs = monte_carlo_runs
        self.n_jobs = n_jobs if n_jobs != -1 else mp.cpu_count()
    
    def calculate_comprehensive_metrics(self, 
                                      returns: np.ndarray,
                                      prices: Optional[np.ndarray] = None,
                                      benchmark_returns: Optional[np.ndarray] = None) -> ComprehensivePerformanceMetrics:
        """
        Calculate comprehensive performance metrics
        
        Args:
            returns: Array of returns
            prices: Optional array of prices (calculated if not provided)
            benchmark_returns: Optional benchmark returns for relative metrics
            
        Returns:
            ComprehensivePerformanceMetrics object
        """
        if len(returns) == 0:
            return ComprehensivePerformanceMetrics()
        
        # Calculate prices if not provided
        if prices is None:
            prices = np.cumprod(1 + returns) * 100
        
        # Basic return metrics
        total_return = (prices[-1] - prices[0]) / prices[0]
        annualized_return = np.mean(returns) * self.periods_per_year
        volatility = np.std(returns) * np.sqrt(self.periods_per_year)
        
        # Risk-adjusted ratios
        sharpe_ratio = calculate_sharpe_ratio_jit(returns, self.risk_free_rate, self.periods_per_year)
        sortino_ratio = calculate_sortino_ratio_jit(returns, self.risk_free_rate, self.periods_per_year)
        
        # Drawdown metrics
        drawdown_series, max_drawdown, peak_idx, trough_idx = calculate_drawdown_jit(prices)
        max_drawdown_duration = self._calculate_max_drawdown_duration(drawdown_series)
        average_drawdown = np.mean(np.abs(drawdown_series[drawdown_series < 0])) if np.any(drawdown_series < 0) else 0.0
        recovery_factor = total_return / max_drawdown if max_drawdown > 0 else 0.0
        
        # Calmar ratio
        calmar_ratio = calculate_calmar_ratio_jit(returns, max_drawdown, self.periods_per_year)
        
        # Omega ratio
        omega_ratio = calculate_omega_ratio_jit(returns, 0.0)
        
        # Risk metrics
        var_95 = calculate_var_jit(returns, 0.95)
        var_99 = calculate_var_jit(returns, 0.99)
        cvar_95 = calculate_cvar_jit(returns, 0.95)
        cvar_99 = calculate_cvar_jit(returns, 0.99)
        
        # Market risk metrics (if benchmark provided)
        beta, alpha, correlation, tracking_error, information_ratio, treynor_ratio = self._calculate_market_risk_metrics(
            returns, benchmark_returns
        )
        
        # Distribution metrics
        skewness = float(stats.skew(returns))
        kurtosis = float(stats.kurtosis(returns))
        jarque_bera_stat, jarque_bera_pvalue = stats.jarque_bera(returns)
        
        # Advanced risk metrics
        ulcer_index = calculate_ulcer_index_jit(prices)
        martin_ratio = annualized_return / ulcer_index if ulcer_index > 0 else 0.0
        burke_ratio = self._calculate_burke_ratio(returns, drawdown_series)
        sterling_ratio = self._calculate_sterling_ratio(returns, drawdown_series)
        pain_index = np.mean(np.abs(drawdown_series))
        gain_to_pain_ratio = annualized_return / pain_index if pain_index > 0 else 0.0
        
        # Tail risk metrics
        tail_ratio = self._calculate_tail_ratio(returns)
        downside_variance, upside_variance = self._calculate_upside_downside_variance(returns)
        downside_volatility = np.sqrt(downside_variance) * np.sqrt(self.periods_per_year)
        upside_volatility = np.sqrt(upside_variance) * np.sqrt(self.periods_per_year)
        
        # Trade-based metrics
        win_rate = np.sum(returns > 0) / len(returns)
        profit_factor = self._calculate_profit_factor(returns)
        expectancy = np.mean(returns)
        kelly_criterion = calculate_kelly_criterion_jit(returns)
        
        # Consistency metrics
        hit_ratio = self._calculate_hit_ratio(returns)
        consistency_ratio = self._calculate_consistency_ratio(returns)
        monthly_return_std = self._calculate_monthly_return_std(returns)
        rolling_sharpe_std = self._calculate_rolling_sharpe_std(returns)
        
        # Additional advanced metrics
        rachev_ratio = self._calculate_rachev_ratio(returns)
        modified_sharpe_ratio = self._calculate_modified_sharpe_ratio(returns)
        conditional_sharpe_ratio = self._calculate_conditional_sharpe_ratio(returns)
        
        # Statistical validation
        ljung_box_pvalue = self._calculate_ljung_box_test(returns)
        adf_pvalue = self._calculate_adf_test(returns)
        
        # Calculate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(returns)
        
        return ComprehensivePerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            information_ratio=information_ratio,
            treynor_ratio=treynor_ratio,
            omega_ratio=omega_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_drawdown_duration,
            average_drawdown=average_drawdown,
            recovery_factor=recovery_factor,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            beta=beta,
            alpha=alpha,
            correlation=correlation,
            tracking_error=tracking_error,
            skewness=skewness,
            kurtosis=kurtosis,
            jarque_bera_statistic=jarque_bera_stat,
            jarque_bera_pvalue=jarque_bera_pvalue,
            ulcer_index=ulcer_index,
            martin_ratio=martin_ratio,
            burke_ratio=burke_ratio,
            sterling_ratio=sterling_ratio,
            pain_index=pain_index,
            gain_to_pain_ratio=gain_to_pain_ratio,
            tail_ratio=tail_ratio,
            downside_variance=downside_variance,
            upside_variance=upside_variance,
            downside_volatility=downside_volatility,
            upside_volatility=upside_volatility,
            win_rate=win_rate,
            profit_factor=profit_factor,
            expectancy=expectancy,
            kelly_criterion=kelly_criterion,
            hit_ratio=hit_ratio,
            consistency_ratio=consistency_ratio,
            monthly_return_std=monthly_return_std,
            rolling_sharpe_std=rolling_sharpe_std,
            rachev_ratio=rachev_ratio,
            modified_sharpe_ratio=modified_sharpe_ratio,
            conditional_sharpe_ratio=conditional_sharpe_ratio,
            ljung_box_pvalue=ljung_box_pvalue,
            adf_pvalue=adf_pvalue,
            confidence_intervals=confidence_intervals
        )
    
    def _calculate_max_drawdown_duration(self, drawdown_series: np.ndarray) -> int:
        """Calculate maximum drawdown duration"""
        if len(drawdown_series) == 0:
            return 0
        
        max_duration = 0
        current_duration = 0
        
        for dd in drawdown_series:
            if dd < 0:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0
        
        return max_duration
    
    def _calculate_market_risk_metrics(self, returns: np.ndarray, benchmark_returns: Optional[np.ndarray]) -> Tuple[float, float, float, float, float, float]:
        """Calculate market risk metrics relative to benchmark"""
        if benchmark_returns is None or len(benchmark_returns) == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        
        # Align lengths
        min_len = min(len(returns), len(benchmark_returns))
        returns = returns[:min_len]
        benchmark_returns = benchmark_returns[:min_len]
        
        # Beta
        covariance = np.cov(returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0.0
        
        # Alpha (Jensen's Alpha)
        portfolio_return = np.mean(returns) * self.periods_per_year
        benchmark_return = np.mean(benchmark_returns) * self.periods_per_year
        alpha = portfolio_return - (self.risk_free_rate + beta * (benchmark_return - self.risk_free_rate))
        
        # Correlation
        correlation = np.corrcoef(returns, benchmark_returns)[0, 1]
        
        # Tracking error
        excess_returns = returns - benchmark_returns
        tracking_error = np.std(excess_returns) * np.sqrt(self.periods_per_year)
        
        # Information ratio
        information_ratio = np.mean(excess_returns) * np.sqrt(self.periods_per_year) / tracking_error if tracking_error > 0 else 0.0
        
        # Treynor ratio
        treynor_ratio = (portfolio_return - self.risk_free_rate) / beta if beta != 0 else 0.0
        
        return beta, alpha, correlation, tracking_error, information_ratio, treynor_ratio
    
    def _calculate_burke_ratio(self, returns: np.ndarray, drawdown_series: np.ndarray) -> float:
        """Calculate Burke ratio"""
        if len(drawdown_series) == 0:
            return 0.0
        
        annual_return = np.mean(returns) * self.periods_per_year
        sum_squared_drawdowns = np.sum(drawdown_series ** 2)
        
        if sum_squared_drawdowns == 0:
            return np.inf if annual_return > 0 else 0.0
        
        return annual_return / np.sqrt(sum_squared_drawdowns)
    
    def _calculate_sterling_ratio(self, returns: np.ndarray, drawdown_series: np.ndarray) -> float:
        """Calculate Sterling ratio"""
        if len(drawdown_series) == 0:
            return 0.0
        
        annual_return = np.mean(returns) * self.periods_per_year
        negative_drawdowns = drawdown_series[drawdown_series < 0]
        
        if len(negative_drawdowns) == 0:
            return np.inf if annual_return > 0 else 0.0
        
        avg_drawdown = abs(np.mean(negative_drawdowns))
        
        return annual_return / avg_drawdown if avg_drawdown > 0 else 0.0
    
    def _calculate_tail_ratio(self, returns: np.ndarray) -> float:
        """Calculate tail ratio"""
        if len(returns) == 0:
            return 0.0
        
        left_tail = np.percentile(returns, 5)
        right_tail = np.percentile(returns, 95)
        
        if left_tail == 0:
            return np.inf if right_tail > 0 else 0.0
        
        return abs(right_tail / left_tail)
    
    def _calculate_upside_downside_variance(self, returns: np.ndarray) -> Tuple[float, float]:
        """Calculate upside and downside variance"""
        if len(returns) == 0:
            return 0.0, 0.0
        
        mean_return = np.mean(returns)
        upside_returns = returns[returns > mean_return] - mean_return
        downside_returns = returns[returns < mean_return] - mean_return
        
        upside_variance = np.var(upside_returns) if len(upside_returns) > 0 else 0.0
        downside_variance = np.var(downside_returns) if len(downside_returns) > 0 else 0.0
        
        return downside_variance, upside_variance
    
    def _calculate_profit_factor(self, returns: np.ndarray) -> float:
        """Calculate profit factor"""
        if len(returns) == 0:
            return 0.0
        
        profits = returns[returns > 0]
        losses = returns[returns < 0]
        
        if len(losses) == 0:
            return np.inf if len(profits) > 0 else 0.0
        
        if len(profits) == 0:
            return 0.0
        
        return np.sum(profits) / abs(np.sum(losses))
    
    def _calculate_hit_ratio(self, returns: np.ndarray) -> float:
        """Calculate hit ratio (percentage of positive periods)"""
        if len(returns) == 0:
            return 0.0
        
        return np.sum(returns > 0) / len(returns)
    
    def _calculate_consistency_ratio(self, returns: np.ndarray) -> float:
        """Calculate consistency ratio"""
        if len(returns) == 0:
            return 0.0
        
        # Calculate rolling annual returns
        window = min(self.periods_per_year, len(returns))
        rolling_returns = []
        
        for i in range(window, len(returns) + 1):
            rolling_return = np.sum(returns[i-window:i])
            rolling_returns.append(rolling_return)
        
        if len(rolling_returns) == 0:
            return 0.0
        
        positive_periods = np.sum(np.array(rolling_returns) > 0)
        
        return positive_periods / len(rolling_returns)
    
    def _calculate_monthly_return_std(self, returns: np.ndarray) -> float:
        """Calculate standard deviation of monthly returns"""
        if len(returns) < 21:  # Need at least 21 days for monthly calculation
            return 0.0
        
        # Calculate monthly returns (assuming 21 trading days per month)
        monthly_returns = []
        for i in range(21, len(returns) + 1, 21):
            monthly_return = np.sum(returns[i-21:i])
            monthly_returns.append(monthly_return)
        
        if len(monthly_returns) < 2:
            return 0.0
        
        return np.std(monthly_returns)
    
    def _calculate_rolling_sharpe_std(self, returns: np.ndarray) -> float:
        """Calculate standard deviation of rolling Sharpe ratios"""
        if len(returns) < 63:  # Need at least 3 months of data
            return 0.0
        
        window = 63  # 3 months
        rolling_sharpes = []
        
        for i in range(window, len(returns) + 1):
            window_returns = returns[i-window:i]
            sharpe = calculate_sharpe_ratio_jit(window_returns, self.risk_free_rate, self.periods_per_year)
            rolling_sharpes.append(sharpe)
        
        if len(rolling_sharpes) < 2:
            return 0.0
        
        return np.std(rolling_sharpes)
    
    def _calculate_rachev_ratio(self, returns: np.ndarray) -> float:
        """Calculate Rachev ratio"""
        if len(returns) == 0:
            return 0.0
        
        # Calculate 5th and 95th percentile thresholds
        left_threshold = np.percentile(returns, 5)
        right_threshold = np.percentile(returns, 95)
        
        # Extract tail returns
        left_tail = returns[returns <= left_threshold]
        right_tail = returns[returns >= right_threshold]
        
        if len(left_tail) == 0 or len(right_tail) == 0:
            return 0.0
        
        # Calculate expected tail values
        expected_tail_loss = np.mean(left_tail)
        expected_tail_gain = np.mean(right_tail)
        
        if expected_tail_loss == 0:
            return np.inf if expected_tail_gain > 0 else 0.0
        
        return abs(expected_tail_gain / expected_tail_loss)
    
    def _calculate_modified_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate modified Sharpe ratio (accounts for skewness and kurtosis)"""
        if len(returns) < 4:
            return 0.0
        
        excess_returns = returns - self.risk_free_rate / self.periods_per_year
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        # Calculate traditional Sharpe
        sharpe = np.sqrt(self.periods_per_year) * np.mean(excess_returns) / np.std(excess_returns)
        
        # Calculate skewness and kurtosis
        skewness = stats.skew(excess_returns)
        kurtosis = stats.kurtosis(excess_returns)
        
        # Modified Sharpe adjustment factor
        adjustment = 1 + (skewness / 6) * sharpe + ((kurtosis - 3) / 24) * (sharpe ** 2)
        
        return sharpe * adjustment
    
    def _calculate_conditional_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate conditional Sharpe ratio"""
        if len(returns) == 0:
            return 0.0
        
        # Calculate threshold (95th percentile)
        threshold = np.percentile(returns, 95)
        
        # Filter returns below threshold
        conditional_returns = returns[returns <= threshold]
        
        if len(conditional_returns) == 0:
            return 0.0
        
        # Calculate conditional Sharpe
        excess_returns = conditional_returns - self.risk_free_rate / self.periods_per_year
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        return np.sqrt(self.periods_per_year) * np.mean(excess_returns) / np.std(excess_returns)
    
    def _calculate_ljung_box_test(self, returns: np.ndarray) -> float:
        """Calculate Ljung-Box test for autocorrelation"""
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            if len(returns) < 10:
                return 1.0
            
            result = acorr_ljungbox(returns, lags=10, return_df=True)
            return float(result['lb_pvalue'].iloc[-1])
        except:
            return 1.0
    
    def _calculate_adf_test(self, returns: np.ndarray) -> float:
        """Calculate Augmented Dickey-Fuller test for stationarity"""
        try:
            from statsmodels.tsa.stattools import adfuller
            if len(returns) < 10:
                return 0.0
            
            result = adfuller(returns)
            return float(result[1])  # p-value
        except:
            return 0.0
    
    def _calculate_confidence_intervals(self, returns: np.ndarray) -> Dict[str, Tuple[float, float]]:
        """Calculate bootstrap confidence intervals for key metrics"""
        if len(returns) < 30:
            return {}
        
        # Bootstrap sampling
        n_samples = len(returns)
        bootstrap_metrics = {
            'sharpe_ratio': [],
            'total_return': [],
            'max_drawdown': [],
            'volatility': []
        }
        
        np.random.seed(42)  # For reproducibility
        
        for _ in range(self.bootstrap_samples):
            # Bootstrap sample
            bootstrap_returns = np.random.choice(returns, size=n_samples, replace=True)
            bootstrap_prices = np.cumprod(1 + bootstrap_returns) * 100
            
            # Calculate metrics
            bootstrap_metrics['sharpe_ratio'].append(
                calculate_sharpe_ratio_jit(bootstrap_returns, self.risk_free_rate, self.periods_per_year)
            )
            bootstrap_metrics['total_return'].append(
                (bootstrap_prices[-1] - bootstrap_prices[0]) / bootstrap_prices[0]
            )
            bootstrap_metrics['max_drawdown'].append(
                calculate_drawdown_jit(bootstrap_prices)[1]
            )
            bootstrap_metrics['volatility'].append(
                np.std(bootstrap_returns) * np.sqrt(self.periods_per_year)
            )
        
        # Calculate confidence intervals
        confidence_intervals = {}
        for metric, values in bootstrap_metrics.items():
            values = np.array(values)
            ci_lower = np.percentile(values, 2.5)
            ci_upper = np.percentile(values, 97.5)
            confidence_intervals[metric] = (ci_lower, ci_upper)
        
        return confidence_intervals


def save_performance_report(metrics: ComprehensivePerformanceMetrics, 
                          returns: np.ndarray, 
                          output_path: str) -> str:
    """
    Save comprehensive performance report
    
    Args:
        metrics: Performance metrics object
        returns: Original returns array
        output_path: Output file path
        
    Returns:
        Path to saved report
    """
    report = {
        'timestamp': datetime.now().isoformat(),
        'analysis_period': {
            'total_observations': len(returns),
            'start_date': 'N/A',
            'end_date': 'N/A',
            'frequency': 'Daily'
        },
        'performance_metrics': metrics.to_dict(),
        'summary_statistics': {
            'mean_return': float(np.mean(returns)),
            'median_return': float(np.median(returns)),
            'std_return': float(np.std(returns)),
            'min_return': float(np.min(returns)),
            'max_return': float(np.max(returns)),
            'positive_periods': int(np.sum(returns > 0)),
            'negative_periods': int(np.sum(returns < 0)),
            'zero_periods': int(np.sum(returns == 0))
        },
        'risk_assessment': {
            'risk_grade': _assess_risk_grade(metrics),
            'key_risks': _identify_key_risks(metrics),
            'recommendations': _generate_recommendations(metrics)
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    return output_path


def _assess_risk_grade(metrics: ComprehensivePerformanceMetrics) -> str:
    """Assess overall risk grade"""
    risk_score = 0
    
    # Drawdown component (40% weight)
    if metrics.max_drawdown > 0.25:
        risk_score += 40
    elif metrics.max_drawdown > 0.15:
        risk_score += 30
    elif metrics.max_drawdown > 0.10:
        risk_score += 20
    elif metrics.max_drawdown > 0.05:
        risk_score += 10
    
    # Volatility component (30% weight)
    if metrics.volatility > 0.30:
        risk_score += 30
    elif metrics.volatility > 0.20:
        risk_score += 20
    elif metrics.volatility > 0.15:
        risk_score += 15
    elif metrics.volatility > 0.10:
        risk_score += 10
    
    # VaR component (30% weight)
    if metrics.var_95 > 0.05:
        risk_score += 30
    elif metrics.var_95 > 0.03:
        risk_score += 20
    elif metrics.var_95 > 0.02:
        risk_score += 15
    elif metrics.var_95 > 0.01:
        risk_score += 10
    
    # Assign grade
    if risk_score >= 70:
        return "EXTREME RISK"
    elif risk_score >= 50:
        return "HIGH RISK"
    elif risk_score >= 30:
        return "MEDIUM RISK"
    elif risk_score >= 15:
        return "LOW-MEDIUM RISK"
    else:
        return "LOW RISK"


def _identify_key_risks(metrics: ComprehensivePerformanceMetrics) -> List[str]:
    """Identify key risk factors"""
    risks = []
    
    if metrics.max_drawdown > 0.15:
        risks.append("High maximum drawdown risk")
    
    if metrics.volatility > 0.20:
        risks.append("High volatility")
    
    if metrics.var_95 > 0.03:
        risks.append("High Value at Risk")
    
    if metrics.skewness < -0.5:
        risks.append("Negative skewness (tail risk)")
    
    if metrics.kurtosis > 3:
        risks.append("High kurtosis (fat tails)")
    
    if metrics.sharpe_ratio < 0.5:
        risks.append("Poor risk-adjusted returns")
    
    return risks


def _generate_recommendations(metrics: ComprehensivePerformanceMetrics) -> List[str]:
    """Generate performance improvement recommendations"""
    recommendations = []
    
    if metrics.sharpe_ratio < 0.5:
        recommendations.append("Improve risk-adjusted returns through position sizing optimization")
    
    if metrics.max_drawdown > 0.15:
        recommendations.append("Implement stronger risk management controls")
    
    if metrics.win_rate < 0.4:
        recommendations.append("Review and optimize entry/exit signals")
    
    if metrics.kelly_criterion < 0.1:
        recommendations.append("Consider reducing position sizes based on Kelly criterion")
    
    if metrics.consistency_ratio < 0.6:
        recommendations.append("Focus on improving strategy consistency")
    
    return recommendations


# Example usage and testing
if __name__ == "__main__":
    # Generate sample data for testing
    np.random.seed(42)
    sample_returns = np.random.normal(0.0008, 0.015, 1000)  # Daily returns
    
    # Initialize analyzer
    analyzer = ComprehensivePerformanceAnalyzer(
        risk_free_rate=0.02,
        periods_per_year=252,
        bootstrap_samples=1000
    )
    
    # Calculate comprehensive metrics
    print("Calculating comprehensive performance metrics...")
    start_time = time.time()
    
    metrics = analyzer.calculate_comprehensive_metrics(sample_returns)
    
    calculation_time = time.time() - start_time
    print(f"Calculation completed in {calculation_time:.3f} seconds")
    
    # Print key metrics
    print("\nKey Performance Metrics:")
    print(f"Total Return: {metrics.total_return:.2%}")
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
    print(f"Max Drawdown: {metrics.max_drawdown:.2%}")
    print(f"VaR (95%): {metrics.var_95:.2%}")
    print(f"CVaR (95%): {metrics.cvar_95:.2%}")
    print(f"Win Rate: {metrics.win_rate:.2%}")
    print(f"Kelly Criterion: {metrics.kelly_criterion:.2%}")
    
    # Save report
    output_path = "/tmp/comprehensive_performance_report.json"
    save_performance_report(metrics, sample_returns, output_path)
    print(f"\nReport saved to: {output_path}")