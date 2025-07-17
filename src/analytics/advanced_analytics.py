#!/usr/bin/env python3
"""
AGENT 4 CRITICAL MISSION: Advanced Analytics and Performance Attribution
=====================================================================

This module provides advanced analytics including:
- Rolling performance analysis
- Regime-based performance evaluation
- Performance attribution analysis
- Factor analysis and correlation studies
- Risk-adjusted performance measures
- Dynamic performance monitoring

Key Features:
- Rolling window analysis with configurable windows
- Regime detection and regime-based performance evaluation
- Performance attribution to various factors
- Dynamic correlation analysis
- Advanced risk decomposition
- Numba JIT optimization for performance
- Comprehensive visualization support

Author: Agent 4 - Performance Analytics Specialist
Date: 2025-07-17
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from scipy import stats
from scipy.signal import find_peaks
from scipy.cluster.hierarchy import linkage, fcluster
import warnings
from numba import jit, njit, prange
from numba.types import float64, int64, boolean
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import time
from datetime import datetime, timedelta
import json
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

warnings.filterwarnings('ignore')


@dataclass
class RollingAnalysisResults:
    """Container for rolling analysis results"""
    
    # Rolling Performance Metrics
    rolling_returns: np.ndarray = field(default_factory=lambda: np.array([]))
    rolling_sharpe: np.ndarray = field(default_factory=lambda: np.array([]))
    rolling_volatility: np.ndarray = field(default_factory=lambda: np.array([]))
    rolling_max_drawdown: np.ndarray = field(default_factory=lambda: np.array([]))
    rolling_win_rate: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Rolling Risk Metrics
    rolling_var: np.ndarray = field(default_factory=lambda: np.array([]))
    rolling_cvar: np.ndarray = field(default_factory=lambda: np.array([]))
    rolling_skewness: np.ndarray = field(default_factory=lambda: np.array([]))
    rolling_kurtosis: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Rolling Correlation
    rolling_correlation: np.ndarray = field(default_factory=lambda: np.array([]))
    rolling_beta: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Window Parameters
    window_size: int = 0
    step_size: int = 1
    
    # Timestamps
    timestamps: np.ndarray = field(default_factory=lambda: np.array([]))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'rolling_returns': self.rolling_returns.tolist(),
            'rolling_sharpe': self.rolling_sharpe.tolist(),
            'rolling_volatility': self.rolling_volatility.tolist(),
            'rolling_max_drawdown': self.rolling_max_drawdown.tolist(),
            'rolling_win_rate': self.rolling_win_rate.tolist(),
            'rolling_var': self.rolling_var.tolist(),
            'rolling_cvar': self.rolling_cvar.tolist(),
            'rolling_skewness': self.rolling_skewness.tolist(),
            'rolling_kurtosis': self.rolling_kurtosis.tolist(),
            'rolling_correlation': self.rolling_correlation.tolist(),
            'rolling_beta': self.rolling_beta.tolist(),
            'window_size': self.window_size,
            'step_size': self.step_size,
            'timestamps': self.timestamps.tolist()
        }


@dataclass
class RegimeAnalysisResults:
    """Container for regime analysis results"""
    
    # Regime Classification
    regime_labels: np.ndarray = field(default_factory=lambda: np.array([]))
    regime_probabilities: np.ndarray = field(default_factory=lambda: np.array([]))
    regime_centers: np.ndarray = field(default_factory=lambda: np.array([]))
    n_regimes: int = 0
    
    # Regime Performance
    regime_performance: Dict[int, Dict[str, float]] = field(default_factory=dict)
    regime_transitions: Dict[Tuple[int, int], int] = field(default_factory=dict)
    
    # Regime Characteristics
    regime_volatility: Dict[int, float] = field(default_factory=dict)
    regime_duration: Dict[int, List[int]] = field(default_factory=dict)
    
    # Regime Stability
    regime_stability_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'regime_labels': self.regime_labels.tolist(),
            'regime_probabilities': self.regime_probabilities.tolist(),
            'regime_centers': self.regime_centers.tolist(),
            'n_regimes': self.n_regimes,
            'regime_performance': self.regime_performance,
            'regime_transitions': {str(k): v for k, v in self.regime_transitions.items()},
            'regime_volatility': self.regime_volatility,
            'regime_duration': {str(k): v for k, v in self.regime_duration.items()},
            'regime_stability_score': self.regime_stability_score
        }


@dataclass
class PerformanceAttributionResults:
    """Container for performance attribution results"""
    
    # Factor Attribution
    factor_loadings: Dict[str, float] = field(default_factory=dict)
    factor_contributions: Dict[str, float] = field(default_factory=dict)
    factor_r_squared: float = 0.0
    
    # Risk Attribution
    systematic_risk: float = 0.0
    idiosyncratic_risk: float = 0.0
    total_risk: float = 0.0
    
    # Return Attribution
    alpha: float = 0.0
    beta_contribution: float = 0.0
    factor_contribution: float = 0.0
    residual_return: float = 0.0
    
    # Factor Analysis
    principal_components: np.ndarray = field(default_factory=lambda: np.array([]))
    explained_variance_ratio: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Time-varying Attribution
    rolling_attribution: Dict[str, np.ndarray] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'factor_loadings': self.factor_loadings,
            'factor_contributions': self.factor_contributions,
            'factor_r_squared': self.factor_r_squared,
            'systematic_risk': self.systematic_risk,
            'idiosyncratic_risk': self.idiosyncratic_risk,
            'total_risk': self.total_risk,
            'alpha': self.alpha,
            'beta_contribution': self.beta_contribution,
            'factor_contribution': self.factor_contribution,
            'residual_return': self.residual_return,
            'principal_components': self.principal_components.tolist(),
            'explained_variance_ratio': self.explained_variance_ratio.tolist(),
            'rolling_attribution': {k: v.tolist() for k, v in self.rolling_attribution.items()}
        }


@njit
def calculate_rolling_sharpe_jit(returns: np.ndarray, window_size: int, risk_free_rate: float, periods_per_year: int) -> np.ndarray:
    """Calculate rolling Sharpe ratio with JIT optimization"""
    if len(returns) < window_size:
        return np.zeros(0, dtype=np.float64)
    
    n_windows = len(returns) - window_size + 1
    rolling_sharpe = np.zeros(n_windows, dtype=np.float64)
    
    for i in range(n_windows):
        window_returns = returns[i:i + window_size]
        excess_returns = window_returns - risk_free_rate / periods_per_year
        
        if np.std(excess_returns) == 0:
            rolling_sharpe[i] = 0.0
        else:
            rolling_sharpe[i] = np.sqrt(periods_per_year) * np.mean(excess_returns) / np.std(excess_returns)
    
    return rolling_sharpe


@njit
def calculate_rolling_volatility_jit(returns: np.ndarray, window_size: int, periods_per_year: int) -> np.ndarray:
    """Calculate rolling volatility with JIT optimization"""
    if len(returns) < window_size:
        return np.zeros(0, dtype=np.float64)
    
    n_windows = len(returns) - window_size + 1
    rolling_volatility = np.zeros(n_windows, dtype=np.float64)
    
    for i in range(n_windows):
        window_returns = returns[i:i + window_size]
        rolling_volatility[i] = np.std(window_returns) * np.sqrt(periods_per_year)
    
    return rolling_volatility


@njit
def calculate_rolling_var_jit(returns: np.ndarray, window_size: int, confidence_level: float) -> np.ndarray:
    """Calculate rolling VaR with JIT optimization"""
    if len(returns) < window_size:
        return np.zeros(0, dtype=np.float64)
    
    n_windows = len(returns) - window_size + 1
    rolling_var = np.zeros(n_windows, dtype=np.float64)
    
    for i in range(n_windows):
        window_returns = returns[i:i + window_size]
        sorted_returns = np.sort(window_returns)
        var_index = int((1 - confidence_level) * len(sorted_returns))
        
        if var_index >= len(sorted_returns):
            var_index = len(sorted_returns) - 1
        
        rolling_var[i] = abs(sorted_returns[var_index])
    
    return rolling_var


@njit
def calculate_rolling_max_drawdown_jit(returns: np.ndarray, window_size: int) -> np.ndarray:
    """Calculate rolling maximum drawdown with JIT optimization"""
    if len(returns) < window_size:
        return np.zeros(0, dtype=np.float64)
    
    n_windows = len(returns) - window_size + 1
    rolling_max_dd = np.zeros(n_windows, dtype=np.float64)
    
    for i in range(n_windows):
        window_returns = returns[i:i + window_size]
        cumulative_returns = np.cumprod(1 + window_returns)
        
        # Calculate running maximum
        running_max = np.zeros_like(cumulative_returns)
        running_max[0] = cumulative_returns[0]
        for j in range(1, len(cumulative_returns)):
            running_max[j] = max(running_max[j-1], cumulative_returns[j])
        
        # Calculate drawdown
        drawdown = (cumulative_returns - running_max) / running_max
        rolling_max_dd[i] = abs(np.min(drawdown))
    
    return rolling_max_dd


class AdvancedAnalytics:
    """
    Advanced analytics engine for comprehensive performance analysis
    """
    
    def __init__(self,
                 risk_free_rate: float = 0.02,
                 periods_per_year: int = 252,
                 confidence_level: float = 0.95,
                 n_jobs: int = -1):
        """
        Initialize advanced analytics engine
        
        Args:
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of periods per year
            confidence_level: Confidence level for VaR calculations
            n_jobs: Number of parallel jobs
        """
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year
        self.confidence_level = confidence_level
        self.n_jobs = n_jobs if n_jobs != -1 else mp.cpu_count()
    
    def calculate_rolling_analysis(self,
                                 returns: np.ndarray,
                                 window_size: int = 252,
                                 step_size: int = 1,
                                 benchmark_returns: Optional[np.ndarray] = None) -> RollingAnalysisResults:
        """
        Calculate comprehensive rolling analysis
        
        Args:
            returns: Array of returns
            window_size: Size of rolling window
            step_size: Step size for rolling window
            benchmark_returns: Optional benchmark returns
            
        Returns:
            RollingAnalysisResults object
        """
        if len(returns) < window_size:
            return RollingAnalysisResults()
        
        print(f"📊 Calculating rolling analysis (window={window_size}, step={step_size})...")
        
        # Calculate rolling performance metrics
        rolling_returns = self._calculate_rolling_returns(returns, window_size, step_size)
        rolling_sharpe = calculate_rolling_sharpe_jit(returns, window_size, self.risk_free_rate, self.periods_per_year)
        rolling_volatility = calculate_rolling_volatility_jit(returns, window_size, self.periods_per_year)
        rolling_max_drawdown = calculate_rolling_max_drawdown_jit(returns, window_size)
        rolling_win_rate = self._calculate_rolling_win_rate(returns, window_size, step_size)
        
        # Calculate rolling risk metrics
        rolling_var = calculate_rolling_var_jit(returns, window_size, self.confidence_level)
        rolling_cvar = self._calculate_rolling_cvar(returns, window_size, step_size)
        rolling_skewness = self._calculate_rolling_skewness(returns, window_size, step_size)
        rolling_kurtosis = self._calculate_rolling_kurtosis(returns, window_size, step_size)
        
        # Calculate rolling correlation and beta (if benchmark provided)
        rolling_correlation = np.array([])
        rolling_beta = np.array([])
        
        if benchmark_returns is not None:
            rolling_correlation = self._calculate_rolling_correlation(returns, benchmark_returns, window_size, step_size)
            rolling_beta = self._calculate_rolling_beta(returns, benchmark_returns, window_size, step_size)
        
        # Generate timestamps
        timestamps = np.arange(len(rolling_returns))
        
        return RollingAnalysisResults(
            rolling_returns=rolling_returns,
            rolling_sharpe=rolling_sharpe,
            rolling_volatility=rolling_volatility,
            rolling_max_drawdown=rolling_max_drawdown,
            rolling_win_rate=rolling_win_rate,
            rolling_var=rolling_var,
            rolling_cvar=rolling_cvar,
            rolling_skewness=rolling_skewness,
            rolling_kurtosis=rolling_kurtosis,
            rolling_correlation=rolling_correlation,
            rolling_beta=rolling_beta,
            window_size=window_size,
            step_size=step_size,
            timestamps=timestamps
        )
    
    def analyze_regimes(self,
                       returns: np.ndarray,
                       n_regimes: int = 3,
                       regime_features: Optional[List[str]] = None) -> RegimeAnalysisResults:
        """
        Analyze market regimes and regime-based performance
        
        Args:
            returns: Array of returns
            n_regimes: Number of regimes to identify
            regime_features: Features to use for regime identification
            
        Returns:
            RegimeAnalysisResults object
        """
        if len(returns) < 100:
            return RegimeAnalysisResults()
        
        print(f"🔍 Analyzing market regimes (n_regimes={n_regimes})...")
        
        # Prepare features for regime identification
        features = self._prepare_regime_features(returns, regime_features)
        
        # Identify regimes using clustering
        regime_labels, regime_centers, regime_probabilities = self._identify_regimes(features, n_regimes)
        
        # Calculate regime performance
        regime_performance = self._calculate_regime_performance(returns, regime_labels, n_regimes)
        
        # Calculate regime transitions
        regime_transitions = self._calculate_regime_transitions(regime_labels, n_regimes)
        
        # Calculate regime characteristics
        regime_volatility = self._calculate_regime_volatility(returns, regime_labels, n_regimes)
        regime_duration = self._calculate_regime_duration(regime_labels, n_regimes)
        
        # Calculate regime stability score
        regime_stability_score = self._calculate_regime_stability(regime_labels, regime_transitions)
        
        return RegimeAnalysisResults(
            regime_labels=regime_labels,
            regime_probabilities=regime_probabilities,
            regime_centers=regime_centers,
            n_regimes=n_regimes,
            regime_performance=regime_performance,
            regime_transitions=regime_transitions,
            regime_volatility=regime_volatility,
            regime_duration=regime_duration,
            regime_stability_score=regime_stability_score
        )
    
    def calculate_performance_attribution(self,
                                        returns: np.ndarray,
                                        factor_returns: Dict[str, np.ndarray],
                                        benchmark_returns: Optional[np.ndarray] = None) -> PerformanceAttributionResults:
        """
        Calculate comprehensive performance attribution
        
        Args:
            returns: Array of strategy returns
            factor_returns: Dictionary of factor returns
            benchmark_returns: Optional benchmark returns
            
        Returns:
            PerformanceAttributionResults object
        """
        if len(returns) < 30:
            return PerformanceAttributionResults()
        
        print("🎯 Calculating performance attribution...")
        
        # Prepare factor data
        factor_data = self._prepare_factor_data(returns, factor_returns)
        
        # Calculate factor loadings and contributions
        factor_loadings, factor_contributions, factor_r_squared = self._calculate_factor_attribution(returns, factor_data)
        
        # Calculate risk attribution
        systematic_risk, idiosyncratic_risk, total_risk = self._calculate_risk_attribution(returns, factor_data, factor_loadings)
        
        # Calculate return attribution
        alpha, beta_contribution, factor_contribution, residual_return = self._calculate_return_attribution(
            returns, factor_data, factor_loadings, benchmark_returns
        )
        
        # Perform principal component analysis
        principal_components, explained_variance_ratio = self._perform_pca_analysis(factor_data)
        
        # Calculate rolling attribution
        rolling_attribution = self._calculate_rolling_attribution(returns, factor_data, factor_loadings)
        
        return PerformanceAttributionResults(
            factor_loadings=factor_loadings,
            factor_contributions=factor_contributions,
            factor_r_squared=factor_r_squared,
            systematic_risk=systematic_risk,
            idiosyncratic_risk=idiosyncratic_risk,
            total_risk=total_risk,
            alpha=alpha,
            beta_contribution=beta_contribution,
            factor_contribution=factor_contribution,
            residual_return=residual_return,
            principal_components=principal_components,
            explained_variance_ratio=explained_variance_ratio,
            rolling_attribution=rolling_attribution
        )
    
    def _calculate_rolling_returns(self, returns: np.ndarray, window_size: int, step_size: int) -> np.ndarray:
        """Calculate rolling returns"""
        if len(returns) < window_size:
            return np.array([])
        
        n_windows = (len(returns) - window_size) // step_size + 1
        rolling_returns = np.zeros(n_windows)
        
        for i in range(n_windows):
            start_idx = i * step_size
            end_idx = start_idx + window_size
            window_returns = returns[start_idx:end_idx]
            rolling_returns[i] = np.prod(1 + window_returns) - 1
        
        return rolling_returns
    
    def _calculate_rolling_win_rate(self, returns: np.ndarray, window_size: int, step_size: int) -> np.ndarray:
        """Calculate rolling win rate"""
        if len(returns) < window_size:
            return np.array([])
        
        n_windows = (len(returns) - window_size) // step_size + 1
        rolling_win_rate = np.zeros(n_windows)
        
        for i in range(n_windows):
            start_idx = i * step_size
            end_idx = start_idx + window_size
            window_returns = returns[start_idx:end_idx]
            rolling_win_rate[i] = np.sum(window_returns > 0) / len(window_returns)
        
        return rolling_win_rate
    
    def _calculate_rolling_cvar(self, returns: np.ndarray, window_size: int, step_size: int) -> np.ndarray:
        """Calculate rolling Conditional VaR"""
        if len(returns) < window_size:
            return np.array([])
        
        n_windows = (len(returns) - window_size) // step_size + 1
        rolling_cvar = np.zeros(n_windows)
        
        for i in range(n_windows):
            start_idx = i * step_size
            end_idx = start_idx + window_size
            window_returns = returns[start_idx:end_idx]
            
            # Calculate VaR threshold
            sorted_returns = np.sort(window_returns)
            var_index = int((1 - self.confidence_level) * len(sorted_returns))
            
            if var_index >= len(sorted_returns):
                var_index = len(sorted_returns) - 1
            
            # Calculate CVaR
            if var_index == 0:
                rolling_cvar[i] = abs(sorted_returns[0])
            else:
                tail_returns = sorted_returns[:var_index]
                rolling_cvar[i] = abs(np.mean(tail_returns))
        
        return rolling_cvar
    
    def _calculate_rolling_skewness(self, returns: np.ndarray, window_size: int, step_size: int) -> np.ndarray:
        """Calculate rolling skewness"""
        if len(returns) < window_size:
            return np.array([])
        
        n_windows = (len(returns) - window_size) // step_size + 1
        rolling_skewness = np.zeros(n_windows)
        
        for i in range(n_windows):
            start_idx = i * step_size
            end_idx = start_idx + window_size
            window_returns = returns[start_idx:end_idx]
            
            if len(window_returns) > 2:
                rolling_skewness[i] = stats.skew(window_returns)
        
        return rolling_skewness
    
    def _calculate_rolling_kurtosis(self, returns: np.ndarray, window_size: int, step_size: int) -> np.ndarray:
        """Calculate rolling kurtosis"""
        if len(returns) < window_size:
            return np.array([])
        
        n_windows = (len(returns) - window_size) // step_size + 1
        rolling_kurtosis = np.zeros(n_windows)
        
        for i in range(n_windows):
            start_idx = i * step_size
            end_idx = start_idx + window_size
            window_returns = returns[start_idx:end_idx]
            
            if len(window_returns) > 3:
                rolling_kurtosis[i] = stats.kurtosis(window_returns)
        
        return rolling_kurtosis
    
    def _calculate_rolling_correlation(self, returns: np.ndarray, benchmark_returns: np.ndarray, window_size: int, step_size: int) -> np.ndarray:
        """Calculate rolling correlation"""
        min_len = min(len(returns), len(benchmark_returns))
        returns = returns[:min_len]
        benchmark_returns = benchmark_returns[:min_len]
        
        if len(returns) < window_size:
            return np.array([])
        
        n_windows = (len(returns) - window_size) // step_size + 1
        rolling_correlation = np.zeros(n_windows)
        
        for i in range(n_windows):
            start_idx = i * step_size
            end_idx = start_idx + window_size
            window_returns = returns[start_idx:end_idx]
            window_benchmark = benchmark_returns[start_idx:end_idx]
            
            correlation = np.corrcoef(window_returns, window_benchmark)[0, 1]
            rolling_correlation[i] = correlation if not np.isnan(correlation) else 0.0
        
        return rolling_correlation
    
    def _calculate_rolling_beta(self, returns: np.ndarray, benchmark_returns: np.ndarray, window_size: int, step_size: int) -> np.ndarray:
        """Calculate rolling beta"""
        min_len = min(len(returns), len(benchmark_returns))
        returns = returns[:min_len]
        benchmark_returns = benchmark_returns[:min_len]
        
        if len(returns) < window_size:
            return np.array([])
        
        n_windows = (len(returns) - window_size) // step_size + 1
        rolling_beta = np.zeros(n_windows)
        
        for i in range(n_windows):
            start_idx = i * step_size
            end_idx = start_idx + window_size
            window_returns = returns[start_idx:end_idx]
            window_benchmark = benchmark_returns[start_idx:end_idx]
            
            covariance = np.cov(window_returns, window_benchmark)[0, 1]
            benchmark_variance = np.var(window_benchmark)
            
            if benchmark_variance > 0:
                rolling_beta[i] = covariance / benchmark_variance
        
        return rolling_beta
    
    def _prepare_regime_features(self, returns: np.ndarray, regime_features: Optional[List[str]]) -> np.ndarray:
        """Prepare features for regime identification"""
        if regime_features is None:
            regime_features = ['returns', 'volatility', 'momentum']
        
        features = []
        
        # Returns feature
        if 'returns' in regime_features:
            features.append(returns)
        
        # Volatility feature (rolling standard deviation)
        if 'volatility' in regime_features:
            window = min(21, len(returns) // 4)  # 21-day rolling volatility
            volatility = pd.Series(returns).rolling(window=window).std().fillna(0).values
            features.append(volatility)
        
        # Momentum feature (rolling mean)
        if 'momentum' in regime_features:
            window = min(21, len(returns) // 4)  # 21-day rolling momentum
            momentum = pd.Series(returns).rolling(window=window).mean().fillna(0).values
            features.append(momentum)
        
        # Skewness feature
        if 'skewness' in regime_features:
            window = min(63, len(returns) // 2)  # 63-day rolling skewness
            skewness = pd.Series(returns).rolling(window=window).skew().fillna(0).values
            features.append(skewness)
        
        # Kurtosis feature
        if 'kurtosis' in regime_features:
            window = min(63, len(returns) // 2)  # 63-day rolling kurtosis
            kurtosis = pd.Series(returns).rolling(window=window).kurt().fillna(0).values
            features.append(kurtosis)
        
        return np.column_stack(features)
    
    def _identify_regimes(self, features: np.ndarray, n_regimes: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Identify regimes using clustering"""
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
        regime_labels = kmeans.fit_predict(features_scaled)
        regime_centers = kmeans.cluster_centers_
        
        # Calculate regime probabilities (distance-based)
        distances = kmeans.transform(features_scaled)
        regime_probabilities = 1 / (1 + distances / np.sum(distances, axis=1, keepdims=True))
        
        return regime_labels, regime_centers, regime_probabilities
    
    def _calculate_regime_performance(self, returns: np.ndarray, regime_labels: np.ndarray, n_regimes: int) -> Dict[int, Dict[str, float]]:
        """Calculate performance metrics for each regime"""
        regime_performance = {}
        
        for regime in range(n_regimes):
            regime_returns = returns[regime_labels == regime]
            
            if len(regime_returns) > 0:
                regime_performance[regime] = {
                    'mean_return': np.mean(regime_returns),
                    'volatility': np.std(regime_returns),
                    'sharpe_ratio': np.mean(regime_returns) / np.std(regime_returns) if np.std(regime_returns) > 0 else 0.0,
                    'win_rate': np.sum(regime_returns > 0) / len(regime_returns),
                    'max_return': np.max(regime_returns),
                    'min_return': np.min(regime_returns),
                    'skewness': float(stats.skew(regime_returns)) if len(regime_returns) > 2 else 0.0,
                    'kurtosis': float(stats.kurtosis(regime_returns)) if len(regime_returns) > 3 else 0.0,
                    'n_observations': len(regime_returns)
                }
        
        return regime_performance
    
    def _calculate_regime_transitions(self, regime_labels: np.ndarray, n_regimes: int) -> Dict[Tuple[int, int], int]:
        """Calculate regime transition matrix"""
        transitions = {}
        
        for i in range(len(regime_labels) - 1):
            current_regime = regime_labels[i]
            next_regime = regime_labels[i + 1]
            
            transition = (current_regime, next_regime)
            transitions[transition] = transitions.get(transition, 0) + 1
        
        return transitions
    
    def _calculate_regime_volatility(self, returns: np.ndarray, regime_labels: np.ndarray, n_regimes: int) -> Dict[int, float]:
        """Calculate volatility for each regime"""
        regime_volatility = {}
        
        for regime in range(n_regimes):
            regime_returns = returns[regime_labels == regime]
            
            if len(regime_returns) > 0:
                regime_volatility[regime] = np.std(regime_returns) * np.sqrt(self.periods_per_year)
        
        return regime_volatility
    
    def _calculate_regime_duration(self, regime_labels: np.ndarray, n_regimes: int) -> Dict[int, List[int]]:
        """Calculate duration of each regime"""
        regime_duration = {regime: [] for regime in range(n_regimes)}
        
        current_regime = regime_labels[0]
        current_duration = 1
        
        for i in range(1, len(regime_labels)):
            if regime_labels[i] == current_regime:
                current_duration += 1
            else:
                regime_duration[current_regime].append(current_duration)
                current_regime = regime_labels[i]
                current_duration = 1
        
        # Add final duration
        regime_duration[current_regime].append(current_duration)
        
        return regime_duration
    
    def _calculate_regime_stability(self, regime_labels: np.ndarray, regime_transitions: Dict[Tuple[int, int], int]) -> float:
        """Calculate regime stability score"""
        if len(regime_labels) < 2:
            return 0.0
        
        # Calculate persistence (diagonal elements of transition matrix)
        total_transitions = sum(regime_transitions.values())
        persistent_transitions = sum(count for (from_regime, to_regime), count in regime_transitions.items() if from_regime == to_regime)
        
        stability_score = persistent_transitions / total_transitions if total_transitions > 0 else 0.0
        
        return stability_score
    
    def _prepare_factor_data(self, returns: np.ndarray, factor_returns: Dict[str, np.ndarray]) -> pd.DataFrame:
        """Prepare factor data for attribution analysis"""
        # Align all series to the same length
        min_length = min(len(returns), min(len(factor_data) for factor_data in factor_returns.values()))
        
        factor_data = pd.DataFrame({
            'returns': returns[:min_length]
        })
        
        for factor_name, factor_values in factor_returns.items():
            factor_data[factor_name] = factor_values[:min_length]
        
        return factor_data
    
    def _calculate_factor_attribution(self, returns: np.ndarray, factor_data: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, float], float]:
        """Calculate factor attribution using linear regression"""
        # Prepare data
        factor_columns = [col for col in factor_data.columns if col != 'returns']
        X = factor_data[factor_columns].values
        y = returns
        
        # Fit linear regression
        reg = LinearRegression()
        reg.fit(X, y)
        
        # Calculate factor loadings
        factor_loadings = {factor_columns[i]: reg.coef_[i] for i in range(len(factor_columns))}
        
        # Calculate factor contributions
        factor_contributions = {}
        for i, factor_name in enumerate(factor_columns):
            factor_mean_return = np.mean(factor_data[factor_name])
            factor_contributions[factor_name] = factor_loadings[factor_name] * factor_mean_return
        
        # Calculate R-squared
        y_pred = reg.predict(X)
        r_squared = r2_score(y, y_pred)
        
        return factor_loadings, factor_contributions, r_squared
    
    def _calculate_risk_attribution(self, returns: np.ndarray, factor_data: pd.DataFrame, factor_loadings: Dict[str, float]) -> Tuple[float, float, float]:
        """Calculate risk attribution"""
        # Calculate factor covariance matrix
        factor_columns = [col for col in factor_data.columns if col != 'returns']
        factor_returns = factor_data[factor_columns].values
        factor_cov = np.cov(factor_returns.T)
        
        # Calculate systematic risk
        loadings_array = np.array(list(factor_loadings.values()))
        systematic_variance = np.dot(loadings_array, np.dot(factor_cov, loadings_array))
        systematic_risk = np.sqrt(systematic_variance) * np.sqrt(self.periods_per_year)
        
        # Calculate total risk
        total_risk = np.std(returns) * np.sqrt(self.periods_per_year)
        
        # Calculate idiosyncratic risk
        idiosyncratic_risk = np.sqrt(max(0, total_risk**2 - systematic_risk**2))
        
        return systematic_risk, idiosyncratic_risk, total_risk
    
    def _calculate_return_attribution(self, returns: np.ndarray, factor_data: pd.DataFrame, factor_loadings: Dict[str, float], benchmark_returns: Optional[np.ndarray]) -> Tuple[float, float, float, float]:
        """Calculate return attribution"""
        # Calculate factor contribution to returns
        factor_columns = [col for col in factor_data.columns if col != 'returns']
        factor_contribution = sum(factor_loadings[factor] * np.mean(factor_data[factor]) for factor in factor_columns)
        
        # Calculate alpha (unexplained return)
        total_return = np.mean(returns) * self.periods_per_year
        alpha = total_return - factor_contribution
        
        # Calculate beta contribution (if benchmark provided)
        beta_contribution = 0.0
        if benchmark_returns is not None and len(benchmark_returns) > 0:
            min_len = min(len(returns), len(benchmark_returns))
            covariance = np.cov(returns[:min_len], benchmark_returns[:min_len])[0, 1]
            benchmark_variance = np.var(benchmark_returns[:min_len])
            
            if benchmark_variance > 0:
                beta = covariance / benchmark_variance
                benchmark_return = np.mean(benchmark_returns[:min_len]) * self.periods_per_year
                beta_contribution = beta * benchmark_return
        
        # Calculate residual return
        residual_return = total_return - factor_contribution - beta_contribution
        
        return alpha, beta_contribution, factor_contribution, residual_return
    
    def _perform_pca_analysis(self, factor_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Perform principal component analysis on factors"""
        factor_columns = [col for col in factor_data.columns if col != 'returns']
        
        if len(factor_columns) < 2:
            return np.array([]), np.array([])
        
        # Standardize data
        scaler = StandardScaler()
        factor_data_scaled = scaler.fit_transform(factor_data[factor_columns])
        
        # Perform PCA
        pca = PCA()
        principal_components = pca.fit_transform(factor_data_scaled)
        explained_variance_ratio = pca.explained_variance_ratio_
        
        return principal_components, explained_variance_ratio
    
    def _calculate_rolling_attribution(self, returns: np.ndarray, factor_data: pd.DataFrame, factor_loadings: Dict[str, float]) -> Dict[str, np.ndarray]:
        """Calculate rolling factor attribution"""
        window_size = min(126, len(returns) // 4)  # 6-month rolling window
        
        if len(returns) < window_size:
            return {}
        
        factor_columns = [col for col in factor_data.columns if col != 'returns']
        rolling_attribution = {factor: [] for factor in factor_columns}
        
        for i in range(window_size, len(returns)):
            window_data = factor_data.iloc[i-window_size:i]
            
            # Calculate rolling factor loadings
            X = window_data[factor_columns].values
            y = returns[i-window_size:i]
            
            try:
                reg = LinearRegression()
                reg.fit(X, y)
                
                for j, factor in enumerate(factor_columns):
                    factor_mean_return = np.mean(window_data[factor])
                    attribution = reg.coef_[j] * factor_mean_return
                    rolling_attribution[factor].append(attribution)
            except:
                # If regression fails, use previous value or zero
                for factor in factor_columns:
                    prev_value = rolling_attribution[factor][-1] if rolling_attribution[factor] else 0.0
                    rolling_attribution[factor].append(prev_value)
        
        # Convert to numpy arrays
        for factor in factor_columns:
            rolling_attribution[factor] = np.array(rolling_attribution[factor])
        
        return rolling_attribution


def generate_comprehensive_analytics_report(returns: np.ndarray,
                                           benchmark_returns: Optional[np.ndarray] = None,
                                           factor_returns: Optional[Dict[str, np.ndarray]] = None,
                                           output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate comprehensive analytics report
    
    Args:
        returns: Array of strategy returns
        benchmark_returns: Optional benchmark returns
        factor_returns: Optional factor returns
        output_path: Optional path to save report
        
    Returns:
        Comprehensive analytics report
    """
    print("📊 Generating comprehensive analytics report...")
    
    # Initialize analytics engine
    analytics = AdvancedAnalytics()
    
    # Calculate rolling analysis
    rolling_results = analytics.calculate_rolling_analysis(returns, benchmark_returns=benchmark_returns)
    
    # Calculate regime analysis
    regime_results = analytics.analyze_regimes(returns)
    
    # Calculate performance attribution (if factor data provided)
    attribution_results = None
    if factor_returns:
        attribution_results = analytics.calculate_performance_attribution(returns, factor_returns, benchmark_returns)
    
    # Compile report
    report = {
        'timestamp': datetime.now().isoformat(),
        'analysis_summary': {
            'total_observations': len(returns),
            'analysis_period': f"{len(returns)} periods",
            'rolling_window_size': rolling_results.window_size,
            'regime_count': regime_results.n_regimes,
            'factor_attribution_available': attribution_results is not None
        },
        'rolling_analysis': rolling_results.to_dict(),
        'regime_analysis': regime_results.to_dict(),
        'performance_attribution': attribution_results.to_dict() if attribution_results else {},
        'summary_insights': _generate_analytics_insights(rolling_results, regime_results, attribution_results)
    }
    
    # Save report if path provided
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"📊 Analytics report saved to: {output_path}")
    
    return report


def _generate_analytics_insights(rolling_results: RollingAnalysisResults,
                                regime_results: RegimeAnalysisResults,
                                attribution_results: Optional[PerformanceAttributionResults]) -> Dict[str, Any]:
    """Generate insights from analytics results"""
    insights = {}
    
    # Rolling analysis insights
    if len(rolling_results.rolling_sharpe) > 0:
        insights['rolling_performance'] = {
            'sharpe_trend': 'improving' if rolling_results.rolling_sharpe[-1] > rolling_results.rolling_sharpe[0] else 'declining',
            'sharpe_volatility': np.std(rolling_results.rolling_sharpe),
            'peak_performance_period': int(np.argmax(rolling_results.rolling_sharpe)),
            'worst_performance_period': int(np.argmin(rolling_results.rolling_sharpe))
        }
    
    # Regime analysis insights
    if regime_results.n_regimes > 0:
        best_regime = max(regime_results.regime_performance.keys(), 
                         key=lambda x: regime_results.regime_performance[x]['sharpe_ratio'])
        worst_regime = min(regime_results.regime_performance.keys(), 
                          key=lambda x: regime_results.regime_performance[x]['sharpe_ratio'])
        
        insights['regime_analysis'] = {
            'best_regime': best_regime,
            'worst_regime': worst_regime,
            'regime_stability': regime_results.regime_stability_score,
            'dominant_regime': int(stats.mode(regime_results.regime_labels)[0][0]) if len(regime_results.regime_labels) > 0 else 0
        }
    
    # Attribution insights
    if attribution_results:
        insights['performance_attribution'] = {
            'explained_variance': attribution_results.factor_r_squared,
            'systematic_risk_proportion': attribution_results.systematic_risk / attribution_results.total_risk if attribution_results.total_risk > 0 else 0,
            'top_factor': max(attribution_results.factor_contributions.keys(), 
                             key=lambda x: abs(attribution_results.factor_contributions[x])) if attribution_results.factor_contributions else None
        }
    
    return insights


# Example usage and testing
if __name__ == "__main__":
    # Generate sample data for testing
    np.random.seed(42)
    sample_returns = np.random.normal(0.0008, 0.015, 1000)  # Daily returns
    benchmark_returns = np.random.normal(0.0005, 0.012, 1000)  # Benchmark returns
    
    # Generate sample factor returns
    factor_returns = {
        'market_factor': np.random.normal(0.0006, 0.013, 1000),
        'size_factor': np.random.normal(0.0002, 0.008, 1000),
        'value_factor': np.random.normal(0.0001, 0.006, 1000)
    }
    
    print("📊 Running comprehensive analytics...")
    
    # Generate comprehensive report
    report = generate_comprehensive_analytics_report(
        returns=sample_returns,
        benchmark_returns=benchmark_returns,
        factor_returns=factor_returns,
        output_path="/tmp/comprehensive_analytics_report.json"
    )
    
    # Print key insights
    print("\n🔍 Key Analytics Insights:")
    if 'rolling_performance' in report['summary_insights']:
        rolling_insights = report['summary_insights']['rolling_performance']
        print(f"  Rolling Sharpe Trend: {rolling_insights['sharpe_trend']}")
        print(f"  Peak Performance Period: {rolling_insights['peak_performance_period']}")
    
    if 'regime_analysis' in report['summary_insights']:
        regime_insights = report['summary_insights']['regime_analysis']
        print(f"  Best Regime: {regime_insights['best_regime']}")
        print(f"  Regime Stability: {regime_insights['regime_stability']:.3f}")
    
    if 'performance_attribution' in report['summary_insights']:
        attribution_insights = report['summary_insights']['performance_attribution']
        print(f"  Explained Variance: {attribution_insights['explained_variance']:.3f}")
        print(f"  Top Factor: {attribution_insights['top_factor']}")
    
    print("\n✅ Advanced analytics completed successfully!")