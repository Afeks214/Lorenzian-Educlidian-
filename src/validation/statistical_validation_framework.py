#!/usr/bin/env python3
"""
AGENT 4 CRITICAL MISSION: Statistical Validation Framework
=========================================================

This module provides comprehensive statistical validation framework with:
- Bootstrap confidence intervals
- Monte Carlo simulation validation
- Hypothesis testing and p-values
- Statistical significance testing
- Robustness testing under different conditions

Key Features:
- Bootstrap resampling for confidence intervals
- Monte Carlo simulation for strategy validation
- Hypothesis testing for statistical significance
- Robustness analysis under various market conditions
- Parallel processing for computational efficiency
- Numba JIT optimization for maximum performance

Author: Agent 4 - Performance Analytics Specialist
Date: 2025-07-17
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from scipy import stats
from scipy.optimize import minimize
import warnings
from numba import jit, njit, prange
from numba.types import float64, int64, boolean
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import time
from datetime import datetime
import json
from pathlib import Path
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
from sklearn.utils import resample

warnings.filterwarnings('ignore')


@dataclass
class StatisticalValidationResults:
    """Container for statistical validation results"""
    
    # Bootstrap Results
    bootstrap_confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    bootstrap_statistics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Monte Carlo Results
    monte_carlo_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    monte_carlo_percentiles: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Hypothesis Testing Results
    hypothesis_tests: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Robustness Testing Results
    robustness_analysis: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Statistical Significance
    significance_tests: Dict[str, bool] = field(default_factory=dict)
    
    # Overall Assessment
    overall_robustness_score: float = 0.0
    statistical_significance_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'bootstrap_confidence_intervals': {k: list(v) for k, v in self.bootstrap_confidence_intervals.items()},
            'bootstrap_statistics': self.bootstrap_statistics,
            'monte_carlo_metrics': self.monte_carlo_metrics,
            'monte_carlo_percentiles': self.monte_carlo_percentiles,
            'hypothesis_tests': self.hypothesis_tests,
            'robustness_analysis': self.robustness_analysis,
            'significance_tests': self.significance_tests,
            'overall_robustness_score': self.overall_robustness_score,
            'statistical_significance_score': self.statistical_significance_score
        }


@njit
def bootstrap_sample_returns(returns: np.ndarray, n_samples: int, seed: int = 42) -> np.ndarray:
    """Generate bootstrap sample of returns with JIT optimization"""
    np.random.seed(seed)
    n_observations = len(returns)
    indices = np.random.randint(0, n_observations, size=n_samples)
    return returns[indices]


@njit
def calculate_bootstrap_sharpe(returns: np.ndarray, risk_free_rate: float, periods_per_year: int) -> float:
    """Calculate Sharpe ratio for bootstrap sample with JIT optimization"""
    if len(returns) == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate / periods_per_year
    
    if np.std(excess_returns) == 0:
        return 0.0
    
    return np.sqrt(periods_per_year) * np.mean(excess_returns) / np.std(excess_returns)


@njit
def calculate_bootstrap_max_drawdown(returns: np.ndarray) -> float:
    """Calculate maximum drawdown for bootstrap sample with JIT optimization"""
    if len(returns) == 0:
        return 0.0
    
    cumulative_returns = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / running_max
    
    return abs(np.min(drawdown))


@njit
def monte_carlo_simulation_single_run(returns: np.ndarray, n_periods: int, seed: int) -> Tuple[float, float, float]:
    """Single Monte Carlo simulation run with JIT optimization"""
    np.random.seed(seed)
    
    # Generate random scenario
    simulated_returns = np.random.choice(returns, size=n_periods, replace=True)
    
    # Calculate metrics
    total_return = np.prod(1 + simulated_returns) - 1
    sharpe_ratio = calculate_bootstrap_sharpe(simulated_returns, 0.02, 252)
    max_drawdown = calculate_bootstrap_max_drawdown(simulated_returns)
    
    return total_return, sharpe_ratio, max_drawdown


class StatisticalValidationFramework:
    """
    Comprehensive statistical validation framework for trading strategies
    """
    
    def __init__(self,
                 bootstrap_samples: int = 10000,
                 monte_carlo_runs: int = 50000,
                 confidence_level: float = 0.95,
                 significance_level: float = 0.05,
                 n_jobs: int = -1,
                 random_seed: int = 42):
        """
        Initialize statistical validation framework
        
        Args:
            bootstrap_samples: Number of bootstrap samples
            monte_carlo_runs: Number of Monte Carlo runs
            confidence_level: Confidence level for intervals
            significance_level: Significance level for tests
            n_jobs: Number of parallel jobs
            random_seed: Random seed for reproducibility
        """
        self.bootstrap_samples = bootstrap_samples
        self.monte_carlo_runs = monte_carlo_runs
        self.confidence_level = confidence_level
        self.significance_level = significance_level
        self.n_jobs = n_jobs if n_jobs != -1 else mp.cpu_count()
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def run_comprehensive_validation(self, returns: np.ndarray) -> StatisticalValidationResults:
        """
        Run comprehensive statistical validation
        
        Args:
            returns: Array of returns
            
        Returns:
            StatisticalValidationResults object
        """
        print("🔬 Starting comprehensive statistical validation...")
        
        # Bootstrap confidence intervals
        print("  📊 Running bootstrap analysis...")
        bootstrap_results = self._run_bootstrap_analysis(returns)
        
        # Monte Carlo simulation
        print("  🎲 Running Monte Carlo simulation...")
        monte_carlo_results = self._run_monte_carlo_simulation(returns)
        
        # Hypothesis testing
        print("  🧪 Running hypothesis tests...")
        hypothesis_results = self._run_hypothesis_tests(returns)
        
        # Robustness analysis
        print("  🛡️ Running robustness analysis...")
        robustness_results = self._run_robustness_analysis(returns)
        
        # Statistical significance assessment
        print("  📈 Assessing statistical significance...")
        significance_results = self._assess_statistical_significance(returns, bootstrap_results, monte_carlo_results)
        
        # Overall robustness score
        robustness_score = self._calculate_robustness_score(bootstrap_results, monte_carlo_results, robustness_results)
        significance_score = self._calculate_significance_score(significance_results)
        
        return StatisticalValidationResults(
            bootstrap_confidence_intervals=bootstrap_results['confidence_intervals'],
            bootstrap_statistics=bootstrap_results['statistics'],
            monte_carlo_metrics=monte_carlo_results['metrics'],
            monte_carlo_percentiles=monte_carlo_results['percentiles'],
            hypothesis_tests=hypothesis_results,
            robustness_analysis=robustness_results,
            significance_tests=significance_results,
            overall_robustness_score=robustness_score,
            statistical_significance_score=significance_score
        )
    
    def _run_bootstrap_analysis(self, returns: np.ndarray) -> Dict[str, Any]:
        """Run bootstrap analysis with confidence intervals"""
        if len(returns) < 30:
            return {'confidence_intervals': {}, 'statistics': {}}
        
        # Metrics to bootstrap
        metrics = ['sharpe_ratio', 'total_return', 'max_drawdown', 'volatility', 'win_rate']
        
        # Parallel bootstrap sampling
        def bootstrap_worker(seed):
            bootstrap_returns = bootstrap_sample_returns(returns, len(returns), seed)
            
            # Calculate metrics
            metrics_dict = {}
            
            # Sharpe ratio
            metrics_dict['sharpe_ratio'] = calculate_bootstrap_sharpe(bootstrap_returns, 0.02, 252)
            
            # Total return
            metrics_dict['total_return'] = np.prod(1 + bootstrap_returns) - 1
            
            # Maximum drawdown
            metrics_dict['max_drawdown'] = calculate_bootstrap_max_drawdown(bootstrap_returns)
            
            # Volatility
            metrics_dict['volatility'] = np.std(bootstrap_returns) * np.sqrt(252)
            
            # Win rate
            metrics_dict['win_rate'] = np.sum(bootstrap_returns > 0) / len(bootstrap_returns)
            
            return metrics_dict
        
        # Run bootstrap samples in parallel
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            seeds = np.random.randint(0, 1000000, self.bootstrap_samples)
            bootstrap_results = list(executor.map(bootstrap_worker, seeds))
        
        # Process results
        bootstrap_metrics = {metric: [] for metric in metrics}
        
        for result in bootstrap_results:
            for metric in metrics:
                bootstrap_metrics[metric].append(result[metric])
        
        # Calculate confidence intervals
        alpha = 1 - self.confidence_level
        confidence_intervals = {}
        statistics = {}
        
        for metric in metrics:
            values = np.array(bootstrap_metrics[metric])
            values = values[~np.isnan(values)]  # Remove NaN values
            values = values[~np.isinf(values)]  # Remove infinite values
            
            if len(values) > 0:
                ci_lower = np.percentile(values, (alpha/2) * 100)
                ci_upper = np.percentile(values, (1 - alpha/2) * 100)
                confidence_intervals[metric] = (ci_lower, ci_upper)
                
                statistics[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'median': np.median(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'skewness': float(stats.skew(values)) if len(values) > 2 else 0.0,
                    'kurtosis': float(stats.kurtosis(values)) if len(values) > 3 else 0.0
                }
        
        return {
            'confidence_intervals': confidence_intervals,
            'statistics': statistics
        }
    
    def _run_monte_carlo_simulation(self, returns: np.ndarray) -> Dict[str, Any]:
        """Run Monte Carlo simulation for strategy validation"""
        if len(returns) < 30:
            return {'metrics': {}, 'percentiles': {}}
        
        # Simulation parameters
        n_periods = len(returns)
        
        # Parallel Monte Carlo simulation
        def monte_carlo_worker(seed):
            return monte_carlo_simulation_single_run(returns, n_periods, seed)
        
        # Run Monte Carlo simulations in parallel
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            seeds = np.random.randint(0, 1000000, self.monte_carlo_runs)
            mc_results = list(executor.map(monte_carlo_worker, seeds))
        
        # Process results
        total_returns = [result[0] for result in mc_results]
        sharpe_ratios = [result[1] for result in mc_results]
        max_drawdowns = [result[2] for result in mc_results]
        
        # Clean data
        total_returns = np.array([x for x in total_returns if not (np.isnan(x) or np.isinf(x))])
        sharpe_ratios = np.array([x for x in sharpe_ratios if not (np.isnan(x) or np.isinf(x))])
        max_drawdowns = np.array([x for x in max_drawdowns if not (np.isnan(x) or np.isinf(x))])
        
        # Calculate statistics
        metrics = {}
        percentiles = {}
        
        for name, values in [('total_return', total_returns), 
                           ('sharpe_ratio', sharpe_ratios), 
                           ('max_drawdown', max_drawdowns)]:
            if len(values) > 0:
                metrics[name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'median': np.median(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
                
                percentiles[name] = {
                    '1st': np.percentile(values, 1),
                    '5th': np.percentile(values, 5),
                    '10th': np.percentile(values, 10),
                    '25th': np.percentile(values, 25),
                    '50th': np.percentile(values, 50),
                    '75th': np.percentile(values, 75),
                    '90th': np.percentile(values, 90),
                    '95th': np.percentile(values, 95),
                    '99th': np.percentile(values, 99)
                }
        
        return {
            'metrics': metrics,
            'percentiles': percentiles
        }
    
    def _run_hypothesis_tests(self, returns: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Run various hypothesis tests"""
        if len(returns) < 10:
            return {}
        
        hypothesis_tests = {}
        
        # Test 1: Normality test (Jarque-Bera)
        try:
            jb_stat, jb_pvalue = stats.jarque_bera(returns)
            hypothesis_tests['normality_jarque_bera'] = {
                'statistic': jb_stat,
                'pvalue': jb_pvalue,
                'is_normal': jb_pvalue > self.significance_level
            }
        except:
            hypothesis_tests['normality_jarque_bera'] = {'statistic': 0.0, 'pvalue': 1.0, 'is_normal': False}
        
        # Test 2: Autocorrelation test (Ljung-Box)
        try:
            if len(returns) >= 20:
                lb_result = acorr_ljungbox(returns, lags=10, return_df=True)
                lb_pvalue = lb_result['lb_pvalue'].iloc[-1]
                hypothesis_tests['autocorrelation_ljung_box'] = {
                    'statistic': lb_result['lb_stat'].iloc[-1],
                    'pvalue': lb_pvalue,
                    'no_autocorrelation': lb_pvalue > self.significance_level
                }
        except:
            hypothesis_tests['autocorrelation_ljung_box'] = {'statistic': 0.0, 'pvalue': 1.0, 'no_autocorrelation': True}
        
        # Test 3: Stationarity test (Augmented Dickey-Fuller)
        try:
            adf_result = adfuller(returns)
            hypothesis_tests['stationarity_adf'] = {
                'statistic': adf_result[0],
                'pvalue': adf_result[1],
                'is_stationary': adf_result[1] < self.significance_level
            }
        except:
            hypothesis_tests['stationarity_adf'] = {'statistic': 0.0, 'pvalue': 1.0, 'is_stationary': False}
        
        # Test 4: Mean reversion test
        try:
            mean_return = np.mean(returns)
            t_stat, t_pvalue = stats.ttest_1samp(returns, 0)
            hypothesis_tests['mean_reversion'] = {
                'statistic': t_stat,
                'pvalue': t_pvalue,
                'mean_significantly_different_from_zero': t_pvalue < self.significance_level
            }
        except:
            hypothesis_tests['mean_reversion'] = {'statistic': 0.0, 'pvalue': 1.0, 'mean_significantly_different_from_zero': False}
        
        # Test 5: Volatility clustering test (ARCH test)
        try:
            squared_returns = returns ** 2
            if len(squared_returns) >= 20:
                lb_result = acorr_ljungbox(squared_returns, lags=10, return_df=True)
                arch_pvalue = lb_result['lb_pvalue'].iloc[-1]
                hypothesis_tests['volatility_clustering_arch'] = {
                    'statistic': lb_result['lb_stat'].iloc[-1],
                    'pvalue': arch_pvalue,
                    'has_volatility_clustering': arch_pvalue < self.significance_level
                }
        except:
            hypothesis_tests['volatility_clustering_arch'] = {'statistic': 0.0, 'pvalue': 1.0, 'has_volatility_clustering': False}
        
        return hypothesis_tests
    
    def _run_robustness_analysis(self, returns: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Run robustness analysis under different conditions"""
        if len(returns) < 100:
            return {}
        
        robustness_results = {}
        
        # Test 1: Subsample stability
        robustness_results['subsample_stability'] = self._test_subsample_stability(returns)
        
        # Test 2: Outlier sensitivity
        robustness_results['outlier_sensitivity'] = self._test_outlier_sensitivity(returns)
        
        # Test 3: Market regime sensitivity
        robustness_results['regime_sensitivity'] = self._test_regime_sensitivity(returns)
        
        # Test 4: Window size sensitivity
        robustness_results['window_sensitivity'] = self._test_window_sensitivity(returns)
        
        return robustness_results
    
    def _test_subsample_stability(self, returns: np.ndarray) -> Dict[str, float]:
        """Test stability across different subsamples"""
        n_subsamples = 10
        subsample_size = len(returns) // 2
        
        subsample_sharpes = []
        subsample_returns = []
        
        for i in range(n_subsamples):
            start_idx = np.random.randint(0, len(returns) - subsample_size)
            subsample = returns[start_idx:start_idx + subsample_size]
            
            # Calculate metrics
            sharpe = calculate_bootstrap_sharpe(subsample, 0.02, 252)
            total_return = np.prod(1 + subsample) - 1
            
            if not (np.isnan(sharpe) or np.isinf(sharpe)):
                subsample_sharpes.append(sharpe)
            if not (np.isnan(total_return) or np.isinf(total_return)):
                subsample_returns.append(total_return)
        
        return {
            'sharpe_stability': np.std(subsample_sharpes) / np.mean(subsample_sharpes) if len(subsample_sharpes) > 0 and np.mean(subsample_sharpes) != 0 else 0.0,
            'return_stability': np.std(subsample_returns) / np.mean(subsample_returns) if len(subsample_returns) > 0 and np.mean(subsample_returns) != 0 else 0.0,
            'n_valid_subsamples': len(subsample_sharpes)
        }
    
    def _test_outlier_sensitivity(self, returns: np.ndarray) -> Dict[str, float]:
        """Test sensitivity to outliers"""
        # Calculate baseline metrics
        baseline_sharpe = calculate_bootstrap_sharpe(returns, 0.02, 252)
        baseline_return = np.prod(1 + returns) - 1
        
        # Remove top and bottom 5% of returns
        trimmed_returns = np.sort(returns)[int(0.05 * len(returns)):int(0.95 * len(returns))]
        
        # Calculate metrics without outliers
        trimmed_sharpe = calculate_bootstrap_sharpe(trimmed_returns, 0.02, 252)
        trimmed_return = np.prod(1 + trimmed_returns) - 1
        
        # Calculate sensitivity
        sharpe_sensitivity = abs(baseline_sharpe - trimmed_sharpe) / abs(baseline_sharpe) if baseline_sharpe != 0 else 0.0
        return_sensitivity = abs(baseline_return - trimmed_return) / abs(baseline_return) if baseline_return != 0 else 0.0
        
        return {
            'sharpe_outlier_sensitivity': sharpe_sensitivity,
            'return_outlier_sensitivity': return_sensitivity,
            'baseline_sharpe': baseline_sharpe,
            'trimmed_sharpe': trimmed_sharpe
        }
    
    def _test_regime_sensitivity(self, returns: np.ndarray) -> Dict[str, float]:
        """Test sensitivity to different market regimes"""
        # Define regimes based on volatility
        rolling_vol = pd.Series(returns).rolling(21).std()
        high_vol_threshold = rolling_vol.quantile(0.7)
        low_vol_threshold = rolling_vol.quantile(0.3)
        
        # Separate returns by regime
        high_vol_returns = returns[rolling_vol > high_vol_threshold]
        low_vol_returns = returns[rolling_vol < low_vol_threshold]
        
        # Calculate metrics for each regime
        high_vol_sharpe = calculate_bootstrap_sharpe(high_vol_returns, 0.02, 252) if len(high_vol_returns) > 10 else 0.0
        low_vol_sharpe = calculate_bootstrap_sharpe(low_vol_returns, 0.02, 252) if len(low_vol_returns) > 10 else 0.0
        
        # Calculate regime sensitivity
        regime_sensitivity = abs(high_vol_sharpe - low_vol_sharpe) if high_vol_sharpe != 0 and low_vol_sharpe != 0 else 0.0
        
        return {
            'regime_sensitivity': regime_sensitivity,
            'high_vol_sharpe': high_vol_sharpe,
            'low_vol_sharpe': low_vol_sharpe,
            'high_vol_periods': len(high_vol_returns),
            'low_vol_periods': len(low_vol_returns)
        }
    
    def _test_window_sensitivity(self, returns: np.ndarray) -> Dict[str, float]:
        """Test sensitivity to different window sizes"""
        window_sizes = [50, 100, 200, len(returns)]
        window_sharpes = []
        
        for window_size in window_sizes:
            if window_size <= len(returns):
                window_returns = returns[-window_size:]
                window_sharpe = calculate_bootstrap_sharpe(window_returns, 0.02, 252)
                if not (np.isnan(window_sharpe) or np.isinf(window_sharpe)):
                    window_sharpes.append(window_sharpe)
        
        # Calculate window sensitivity
        window_sensitivity = np.std(window_sharpes) / np.mean(window_sharpes) if len(window_sharpes) > 0 and np.mean(window_sharpes) != 0 else 0.0
        
        return {
            'window_sensitivity': window_sensitivity,
            'window_sharpe_std': np.std(window_sharpes) if len(window_sharpes) > 0 else 0.0,
            'window_sharpe_mean': np.mean(window_sharpes) if len(window_sharpes) > 0 else 0.0
        }
    
    def _assess_statistical_significance(self, returns: np.ndarray, bootstrap_results: Dict, monte_carlo_results: Dict) -> Dict[str, bool]:
        """Assess statistical significance of key metrics"""
        significance_results = {}
        
        # Sharpe ratio significance
        if 'sharpe_ratio' in bootstrap_results['confidence_intervals']:
            ci_lower, ci_upper = bootstrap_results['confidence_intervals']['sharpe_ratio']
            significance_results['sharpe_ratio_significant'] = ci_lower > 0.0
        
        # Return significance
        if 'total_return' in bootstrap_results['confidence_intervals']:
            ci_lower, ci_upper = bootstrap_results['confidence_intervals']['total_return']
            significance_results['return_significant'] = ci_lower > 0.0
        
        # Drawdown significance
        if 'max_drawdown' in bootstrap_results['confidence_intervals']:
            ci_lower, ci_upper = bootstrap_results['confidence_intervals']['max_drawdown']
            significance_results['drawdown_significant'] = ci_upper < 0.2  # Less than 20% drawdown
        
        # Win rate significance
        if 'win_rate' in bootstrap_results['confidence_intervals']:
            ci_lower, ci_upper = bootstrap_results['confidence_intervals']['win_rate']
            significance_results['win_rate_significant'] = ci_lower > 0.5
        
        # Monte Carlo consistency
        if 'sharpe_ratio' in monte_carlo_results['metrics']:
            mc_sharpe_std = monte_carlo_results['metrics']['sharpe_ratio']['std']
            significance_results['monte_carlo_consistent'] = mc_sharpe_std < 0.5
        
        return significance_results
    
    def _calculate_robustness_score(self, bootstrap_results: Dict, monte_carlo_results: Dict, robustness_results: Dict) -> float:
        """Calculate overall robustness score"""
        scores = []
        
        # Bootstrap consistency score
        if 'sharpe_ratio' in bootstrap_results['statistics']:
            bootstrap_std = bootstrap_results['statistics']['sharpe_ratio']['std']
            bootstrap_mean = bootstrap_results['statistics']['sharpe_ratio']['mean']
            if bootstrap_mean != 0:
                cv = bootstrap_std / abs(bootstrap_mean)
                scores.append(max(0, 1 - cv))  # Lower coefficient of variation = higher score
        
        # Monte Carlo consistency score
        if 'sharpe_ratio' in monte_carlo_results['metrics']:
            mc_std = monte_carlo_results['metrics']['sharpe_ratio']['std']
            mc_mean = monte_carlo_results['metrics']['sharpe_ratio']['mean']
            if mc_mean != 0:
                cv = mc_std / abs(mc_mean)
                scores.append(max(0, 1 - cv))
        
        # Robustness test scores
        if 'subsample_stability' in robustness_results:
            stability_score = robustness_results['subsample_stability']['sharpe_stability']
            scores.append(max(0, 1 - stability_score))
        
        if 'outlier_sensitivity' in robustness_results:
            sensitivity_score = robustness_results['outlier_sensitivity']['sharpe_outlier_sensitivity']
            scores.append(max(0, 1 - sensitivity_score))
        
        return np.mean(scores) if scores else 0.0
    
    def _calculate_significance_score(self, significance_results: Dict[str, bool]) -> float:
        """Calculate overall statistical significance score"""
        if not significance_results:
            return 0.0
        
        significant_count = sum(significance_results.values())
        total_tests = len(significance_results)
        
        return significant_count / total_tests


def run_comprehensive_statistical_validation(returns: np.ndarray, output_path: Optional[str] = None) -> StatisticalValidationResults:
    """
    Run comprehensive statistical validation analysis
    
    Args:
        returns: Array of returns
        output_path: Optional path to save results
        
    Returns:
        StatisticalValidationResults object
    """
    # Initialize framework
    framework = StatisticalValidationFramework(
        bootstrap_samples=10000,
        monte_carlo_runs=50000,
        confidence_level=0.95,
        significance_level=0.05
    )
    
    # Run validation
    start_time = time.time()
    results = framework.run_comprehensive_validation(returns)
    validation_time = time.time() - start_time
    
    print(f"✅ Statistical validation completed in {validation_time:.2f} seconds")
    
    # Save results if path provided
    if output_path:
        validation_report = {
            'timestamp': datetime.now().isoformat(),
            'validation_time_seconds': validation_time,
            'framework_parameters': {
                'bootstrap_samples': framework.bootstrap_samples,
                'monte_carlo_runs': framework.monte_carlo_runs,
                'confidence_level': framework.confidence_level,
                'significance_level': framework.significance_level
            },
            'results': results.to_dict(),
            'summary': {
                'overall_robustness_score': results.overall_robustness_score,
                'statistical_significance_score': results.statistical_significance_score,
                'assessment': _generate_validation_assessment(results)
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(validation_report, f, indent=2)
        
        print(f"📊 Validation report saved to: {output_path}")
    
    return results


def _generate_validation_assessment(results: StatisticalValidationResults) -> str:
    """Generate overall validation assessment"""
    robustness_score = results.overall_robustness_score
    significance_score = results.statistical_significance_score
    
    if robustness_score > 0.8 and significance_score > 0.8:
        return "EXCELLENT - Strategy is highly robust and statistically significant"
    elif robustness_score > 0.6 and significance_score > 0.6:
        return "GOOD - Strategy shows good robustness and statistical validity"
    elif robustness_score > 0.4 and significance_score > 0.4:
        return "MODERATE - Strategy has acceptable statistical properties"
    else:
        return "POOR - Strategy lacks statistical robustness and significance"


# Example usage and testing
if __name__ == "__main__":
    # Generate sample data for testing
    np.random.seed(42)
    sample_returns = np.random.normal(0.0008, 0.015, 1000)  # Daily returns
    
    print("🔬 Running comprehensive statistical validation...")
    
    # Run validation
    results = run_comprehensive_statistical_validation(
        returns=sample_returns,
        output_path="/tmp/statistical_validation_report.json"
    )
    
    # Print key results
    print("\n📊 Statistical Validation Results:")
    print(f"Overall Robustness Score: {results.overall_robustness_score:.3f}")
    print(f"Statistical Significance Score: {results.statistical_significance_score:.3f}")
    
    # Print confidence intervals
    print("\n📈 Bootstrap Confidence Intervals:")
    for metric, (lower, upper) in results.bootstrap_confidence_intervals.items():
        print(f"  {metric}: [{lower:.4f}, {upper:.4f}]")
    
    # Print hypothesis test results
    print("\n🧪 Hypothesis Test Results:")
    for test_name, test_result in results.hypothesis_tests.items():
        print(f"  {test_name}: p-value = {test_result['pvalue']:.4f}")
    
    print("\n✅ Statistical validation completed successfully!")