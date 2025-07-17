"""
Alpha Validation for Strategic MARL System

This module provides the final validation that the MARL system generates
alpha (excess returns) compared to baseline strategies.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Callable, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.utils import resample
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.sandwich_covariance import cov_hac
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.multitest import multipletests
from scipy.stats import norm
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from .run_backtest import BacktestRunner, BacktestConfig, BacktestResult
from .metrics import compare_performance

logger = logging.getLogger(__name__)


class AlphaValidator:
    """
    Validates that MARL system generates statistically significant alpha
    """
    
    def __init__(self, significance_level: float = 0.05, n_bootstrap: int = 1000, block_size: int = 20):
        """
        Initialize alpha validator
        
        Args:
            significance_level: Statistical significance level for tests
            n_bootstrap: Number of bootstrap samples for confidence intervals
            block_size: Block size for block bootstrap (for time series)
        """
        self.significance_level = significance_level
        self.n_bootstrap = n_bootstrap
        self.block_size = block_size
        self.results: Dict[str, Any] = {}
        self.multiple_testing_methods = ['bonferroni', 'fdr_bh', 'fdr_by', 'holm']
        
    def validate_alpha(
        self,
        marl_result: BacktestResult,
        baseline_result: BacktestResult
    ) -> Dict[str, Any]:
        """
        Validate MARL alpha generation vs baseline
        
        Args:
            marl_result: MARL backtest result
            baseline_result: Baseline backtest result
            
        Returns:
            Dictionary with validation results
        """
        validation = {}
        
        # 1. Performance comparison
        perf_comparison = self._compare_performance(marl_result, baseline_result)
        validation['performance'] = perf_comparison
        
        # 2. Statistical tests
        stat_tests = self._run_statistical_tests(
            marl_result.returns,
            baseline_result.returns
        )
        validation['statistical_tests'] = stat_tests
        
        # 3. Risk-adjusted alpha
        risk_adjusted = self._calculate_risk_adjusted_alpha(
            marl_result,
            baseline_result
        )
        validation['risk_adjusted_alpha'] = risk_adjusted
        
        # 4. Bootstrap confidence intervals
        bootstrap_results = self._bootstrap_analysis(marl_result, baseline_result)
        validation['bootstrap_analysis'] = bootstrap_results
        
        # 5. Advanced bootstrap methods (BCa, studentized)
        advanced_bootstrap = self._advanced_bootstrap_methods(marl_result, baseline_result)
        validation['advanced_bootstrap'] = advanced_bootstrap
        
        # 6. Permutation tests
        permutation_results = self._permutation_tests(marl_result, baseline_result)
        validation['permutation_tests'] = permutation_results
        
        # 7. HAC standard errors
        hac_results = self._hac_standard_errors(marl_result, baseline_result)
        validation['hac_analysis'] = hac_results
        
        # 8. Power analysis
        power_analysis = self._power_analysis(marl_result, baseline_result)
        validation['power_analysis'] = power_analysis
        
        # 9. Multiple testing corrections
        multiple_testing = self._multiple_testing_corrections(validation)
        validation['multiple_testing'] = multiple_testing
        
        # 10. Consistency analysis
        consistency = self._analyze_consistency(marl_result, baseline_result)
        validation['consistency'] = consistency
        
        # 11. Overall verdict
        validation['verdict'] = self._generate_verdict(validation)
        
        self.results = validation
        return validation
        
    def _compare_performance(
        self,
        marl_result: BacktestResult,
        baseline_result: BacktestResult
    ) -> Dict[str, float]:
        """Compare raw performance metrics"""
        comparison = {}
        
        # Key metrics
        metrics_to_compare = [
            'total_return', 'sharpe_ratio', 'max_drawdown',
            'win_rate', 'profit_factor', 'calmar_ratio'
        ]
        
        for metric in metrics_to_compare:
            marl_val = getattr(marl_result.metrics, metric)
            base_val = getattr(baseline_result.metrics, metric)
            
            if base_val != 0:
                if metric == 'max_drawdown':  # Lower is better
                    improvement = (base_val - marl_val) / abs(base_val)
                else:  # Higher is better
                    improvement = (marl_val - base_val) / abs(base_val)
            else:
                improvement = float('inf') if marl_val > 0 else 0.0
                
            comparison[f'{metric}_improvement'] = improvement
            comparison[f'{metric}_marl'] = marl_val
            comparison[f'{metric}_baseline'] = base_val
            
        return comparison
        
    def _run_statistical_tests(
        self,
        marl_returns: np.ndarray,
        baseline_returns: np.ndarray
    ) -> Dict[str, Any]:
        """Run statistical significance tests"""
        results = {}
        
        # Ensure same length
        min_len = min(len(marl_returns), len(baseline_returns))
        marl_returns = marl_returns[:min_len]
        baseline_returns = baseline_returns[:min_len]
        
        # 1. T-test for mean returns
        t_stat, t_pvalue = stats.ttest_ind(marl_returns, baseline_returns)
        results['ttest'] = {
            'statistic': t_stat,
            'pvalue': t_pvalue,
            'significant': t_pvalue < self.significance_level,
            'marl_mean': np.mean(marl_returns),
            'baseline_mean': np.mean(baseline_returns)
        }
        
        # 2. Mann-Whitney U test (non-parametric)
        u_stat, u_pvalue = stats.mannwhitneyu(
            marl_returns, baseline_returns, alternative='greater'
        )
        results['mann_whitney'] = {
            'statistic': u_stat,
            'pvalue': u_pvalue,
            'significant': u_pvalue < self.significance_level
        }
        
        # 3. Levene's test for variance equality
        levene_stat, levene_pvalue = stats.levene(marl_returns, baseline_returns)
        results['levene'] = {
            'statistic': levene_stat,
            'pvalue': levene_pvalue,
            'equal_variance': levene_pvalue > self.significance_level,
            'marl_std': np.std(marl_returns),
            'baseline_std': np.std(baseline_returns)
        }
        
        # 4. Paired t-test (if returns are paired)
        excess_returns = marl_returns - baseline_returns
        paired_t_stat, paired_t_pvalue = stats.ttest_1samp(excess_returns, 0)
        results['paired_ttest'] = {
            'statistic': paired_t_stat,
            'pvalue': paired_t_pvalue,
            'significant': paired_t_pvalue < self.significance_level,
            'mean_excess': np.mean(excess_returns)
        }
        
        return results
        
    def _bootstrap_analysis(
        self,
        marl_result: BacktestResult,
        baseline_result: BacktestResult
    ) -> Dict[str, Any]:
        """
        Bootstrap analysis for confidence intervals and robust statistics
        """
        bootstrap_results = {}
        
        # Prepare data
        min_len = min(len(marl_result.returns), len(baseline_result.returns))
        marl_returns = marl_result.returns[:min_len]
        baseline_returns = baseline_result.returns[:min_len]
        excess_returns = marl_returns - baseline_returns
        
        # 1. Bootstrap confidence intervals for alpha
        alpha_bootstrap = []
        sharpe_bootstrap = []
        info_ratio_bootstrap = []
        
        for _ in range(self.n_bootstrap):
            # Resample with replacement
            bootstrap_indices = resample(range(len(excess_returns)), n_samples=len(excess_returns))
            bootstrap_excess = excess_returns[bootstrap_indices]
            bootstrap_marl = marl_returns[bootstrap_indices]
            
            # Calculate metrics
            alpha_bootstrap.append(np.mean(bootstrap_excess))
            
            if np.std(bootstrap_marl) > 0:
                sharpe_bootstrap.append(np.mean(bootstrap_marl) / np.std(bootstrap_marl) * np.sqrt(252 * 78))
            else:
                sharpe_bootstrap.append(0.0)
                
            if np.std(bootstrap_excess) > 0:
                info_ratio_bootstrap.append(np.mean(bootstrap_excess) / np.std(bootstrap_excess) * np.sqrt(252 * 78))
            else:
                info_ratio_bootstrap.append(0.0)
        
        # Calculate confidence intervals
        alpha_ci = np.percentile(alpha_bootstrap, [2.5, 97.5])
        sharpe_ci = np.percentile(sharpe_bootstrap, [2.5, 97.5])
        info_ratio_ci = np.percentile(info_ratio_bootstrap, [2.5, 97.5])
        
        bootstrap_results['alpha'] = {
            'mean': np.mean(alpha_bootstrap),
            'std': np.std(alpha_bootstrap),
            'confidence_interval': alpha_ci,
            'significant': not (alpha_ci[0] <= 0 <= alpha_ci[1])
        }
        
        bootstrap_results['sharpe_ratio'] = {
            'mean': np.mean(sharpe_bootstrap),
            'std': np.std(sharpe_bootstrap),
            'confidence_interval': sharpe_ci
        }
        
        bootstrap_results['information_ratio'] = {
            'mean': np.mean(info_ratio_bootstrap),
            'std': np.std(info_ratio_bootstrap),
            'confidence_interval': info_ratio_ci,
            'significant': not (info_ratio_ci[0] <= 0 <= info_ratio_ci[1])
        }
        
        # 2. Block bootstrap for time series
        block_bootstrap_results = self._block_bootstrap(excess_returns)
        bootstrap_results['block_bootstrap'] = block_bootstrap_results
        
        return bootstrap_results
        
    def _advanced_bootstrap_methods(
        self,
        marl_result: BacktestResult,
        baseline_result: BacktestResult
    ) -> Dict[str, Any]:
        """
        Advanced bootstrap methods: BCa (Bias-Corrected and Accelerated) and studentized bootstrap
        """
        advanced_results = {}
        
        # Prepare data
        min_len = min(len(marl_result.returns), len(baseline_result.returns))
        marl_returns = marl_result.returns[:min_len]
        baseline_returns = baseline_result.returns[:min_len]
        excess_returns = marl_returns - baseline_returns
        
        # 1. BCa Bootstrap for alpha
        alpha_estimate = np.mean(excess_returns)
        bca_ci = self._bca_bootstrap(excess_returns, np.mean, alpha_estimate)
        
        advanced_results['bca_alpha'] = {
            'confidence_interval': bca_ci,
            'estimate': alpha_estimate,
            'significant': not (bca_ci[0] <= 0 <= bca_ci[1])
        }
        
        # 2. Studentized Bootstrap for alpha
        studentized_ci = self._studentized_bootstrap(excess_returns, np.mean)
        
        advanced_results['studentized_alpha'] = {
            'confidence_interval': studentized_ci,
            'estimate': alpha_estimate,
            'significant': not (studentized_ci[0] <= 0 <= studentized_ci[1])
        }
        
        # 3. BCa Bootstrap for Sharpe ratio
        sharpe_estimate = np.mean(marl_returns) / np.std(marl_returns) * np.sqrt(252 * 78) if np.std(marl_returns) > 0 else 0
        
        def sharpe_func(x):
            return np.mean(x) / np.std(x) * np.sqrt(252 * 78) if np.std(x) > 0 else 0
            
        sharpe_bca_ci = self._bca_bootstrap(marl_returns, sharpe_func, sharpe_estimate)
        
        advanced_results['bca_sharpe'] = {
            'confidence_interval': sharpe_bca_ci,
            'estimate': sharpe_estimate
        }
        
        # 4. Studentized Bootstrap for Information Ratio
        info_ratio_estimate = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252 * 78) if np.std(excess_returns) > 0 else 0
        
        def info_ratio_func(x):
            return np.mean(x) / np.std(x) * np.sqrt(252 * 78) if np.std(x) > 0 else 0
            
        info_ratio_studentized_ci = self._studentized_bootstrap(excess_returns, info_ratio_func)
        
        advanced_results['studentized_info_ratio'] = {
            'confidence_interval': info_ratio_studentized_ci,
            'estimate': info_ratio_estimate,
            'significant': not (info_ratio_studentized_ci[0] <= 0 <= info_ratio_studentized_ci[1])
        }
        
        return advanced_results
        
    def _bca_bootstrap(self, data: np.ndarray, statistic: Callable, original_stat: float) -> Tuple[float, float]:
        """
        Bias-Corrected and Accelerated (BCa) bootstrap confidence interval
        """
        n = len(data)
        
        # Bootstrap samples
        bootstrap_stats = []
        for _ in range(self.n_bootstrap):
            bootstrap_sample = resample(data, n_samples=n)
            bootstrap_stats.append(statistic(bootstrap_sample))
        
        bootstrap_stats = np.array(bootstrap_stats)
        
        # Bias correction
        bias_correction = norm.ppf(np.mean(bootstrap_stats < original_stat))
        
        # Acceleration (jackknife)
        jackknife_stats = []
        for i in range(n):
            jackknife_sample = np.concatenate([data[:i], data[i+1:]])
            jackknife_stats.append(statistic(jackknife_sample))
        
        jackknife_stats = np.array(jackknife_stats)
        jackknife_mean = np.mean(jackknife_stats)
        
        # Calculate acceleration
        numerator = np.sum((jackknife_mean - jackknife_stats) ** 3)
        denominator = 6 * (np.sum((jackknife_mean - jackknife_stats) ** 2) ** 1.5)
        acceleration = numerator / denominator if denominator != 0 else 0
        
        # Calculate adjusted percentiles
        alpha = (1 - 0.95) / 2  # For 95% CI
        z_alpha = norm.ppf(alpha)
        z_1_alpha = norm.ppf(1 - alpha)
        
        # BCa adjusted percentiles
        alpha_1 = norm.cdf(bias_correction + (bias_correction + z_alpha) / (1 - acceleration * (bias_correction + z_alpha)))
        alpha_2 = norm.cdf(bias_correction + (bias_correction + z_1_alpha) / (1 - acceleration * (bias_correction + z_1_alpha)))
        
        # Ensure percentiles are within bounds
        alpha_1 = max(0, min(1, alpha_1))
        alpha_2 = max(0, min(1, alpha_2))
        
        # Return confidence interval
        lower = np.percentile(bootstrap_stats, alpha_1 * 100)
        upper = np.percentile(bootstrap_stats, alpha_2 * 100)
        
        return (lower, upper)
        
    def _studentized_bootstrap(self, data: np.ndarray, statistic: Callable) -> Tuple[float, float]:
        """
        Studentized bootstrap confidence interval
        """
        n = len(data)
        original_stat = statistic(data)
        
        # First-level bootstrap
        bootstrap_stats = []
        studentized_stats = []
        
        for _ in range(self.n_bootstrap):
            bootstrap_sample = resample(data, n_samples=n)
            bootstrap_stat = statistic(bootstrap_sample)
            bootstrap_stats.append(bootstrap_stat)
            
            # Second-level bootstrap for standard error estimation
            second_level_stats = []
            for _ in range(min(100, self.n_bootstrap // 10)):  # Reduced iterations for speed
                second_bootstrap = resample(bootstrap_sample, n_samples=n)
                second_level_stats.append(statistic(second_bootstrap))
            
            se = np.std(second_level_stats)
            if se > 0:
                studentized_stats.append((bootstrap_stat - original_stat) / se)
            else:
                studentized_stats.append(0)
        
        studentized_stats = np.array(studentized_stats)
        
        # Calculate standard error of original statistic
        original_se = np.std(bootstrap_stats)
        
        # Get critical values
        alpha = (1 - 0.95) / 2  # For 95% CI
        t_alpha = np.percentile(studentized_stats, alpha * 100)
        t_1_alpha = np.percentile(studentized_stats, (1 - alpha) * 100)
        
        # Construct confidence interval
        lower = original_stat - t_1_alpha * original_se
        upper = original_stat - t_alpha * original_se
        
        return (lower, upper)
        
    def _power_analysis(
        self,
        marl_result: BacktestResult,
        baseline_result: BacktestResult
    ) -> Dict[str, Any]:
        """
        Power analysis for statistical tests
        """
        power_results = {}
        
        # Prepare data
        min_len = min(len(marl_result.returns), len(baseline_result.returns))
        marl_returns = marl_result.returns[:min_len]
        baseline_returns = baseline_result.returns[:min_len]
        excess_returns = marl_returns - baseline_returns
        
        # Observed effect size (Cohen's d)
        pooled_std = np.sqrt(((len(marl_returns) - 1) * np.var(marl_returns) + 
                             (len(baseline_returns) - 1) * np.var(baseline_returns)) / 
                            (len(marl_returns) + len(baseline_returns) - 2))
        
        effect_size = (np.mean(marl_returns) - np.mean(baseline_returns)) / pooled_std if pooled_std > 0 else 0
        
        power_results['effect_size'] = effect_size
        
        # Power for current sample size
        current_power = self._calculate_power(effect_size, min_len, self.significance_level)
        power_results['current_power'] = current_power
        
        # Required sample size for different power levels
        target_powers = [0.8, 0.9, 0.95]
        required_n = {}
        
        for target_power in target_powers:
            n_required = self._calculate_required_n(effect_size, target_power, self.significance_level)
            required_n[f'power_{target_power}'] = n_required
        
        power_results['required_sample_sizes'] = required_n
        
        # Minimum detectable effect size
        min_effect_size = self._calculate_min_effect_size(min_len, 0.8, self.significance_level)
        power_results['min_detectable_effect_size'] = min_effect_size
        
        return power_results
        
    def _calculate_power(self, effect_size: float, n: int, alpha: float) -> float:
        """
        Calculate statistical power for two-sample t-test
        """
        if effect_size == 0:
            return alpha
        
        # Critical value
        t_critical = stats.t.ppf(1 - alpha/2, 2*n - 2)
        
        # Non-centrality parameter
        ncp = effect_size * np.sqrt(n/2)
        
        # Power calculation
        power = 1 - stats.t.cdf(t_critical, 2*n - 2, ncp) + stats.t.cdf(-t_critical, 2*n - 2, ncp)
        
        return power
        
    def _calculate_required_n(self, effect_size: float, target_power: float, alpha: float) -> int:
        """
        Calculate required sample size for target power
        """
        if effect_size == 0:
            return float('inf')
        
        def power_func(n):
            return self._calculate_power(effect_size, int(n), alpha) - target_power
        
        try:
            result = minimize_scalar(lambda n: abs(power_func(n)), bounds=(10, 10000), method='bounded')
            return int(result.x)
        except (ValueError, TypeError, AttributeError, KeyError) as e:
            return 10000  # Default large value if optimization fails
            
    def _calculate_min_effect_size(self, n: int, target_power: float, alpha: float) -> float:
        """
        Calculate minimum detectable effect size
        """
        def power_func(effect_size):
            return self._calculate_power(effect_size, n, alpha) - target_power
        
        try:
            result = minimize_scalar(lambda es: abs(power_func(es)), bounds=(0, 3), method='bounded')
            return result.x
        except (ValueError, TypeError, AttributeError, KeyError) as e:
            return 0.5  # Default moderate effect size
            
    def _multiple_testing_corrections(self, validation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply multiple testing corrections to p-values
        """
        corrections = {}
        
        # Collect p-values from different tests
        p_values = []
        test_names = []
        
        # Statistical tests
        stat_tests = validation.get('statistical_tests', {})
        for test_name, test_result in stat_tests.items():
            if isinstance(test_result, dict) and 'pvalue' in test_result:
                p_values.append(test_result['pvalue'])
                test_names.append(f'stat_{test_name}')
        
        # HAC analysis
        hac_analysis = validation.get('hac_analysis', {})
        if 'alpha_test' in hac_analysis and 'p_value' in hac_analysis['alpha_test']:
            p_values.append(hac_analysis['alpha_test']['p_value'])
            test_names.append('hac_alpha')
        
        # Permutation tests
        perm_tests = validation.get('permutation_tests', {})
        for test_name, test_result in perm_tests.items():
            if isinstance(test_result, dict) and 'p_value' in test_result:
                p_values.append(test_result['p_value'])
                test_names.append(f'perm_{test_name}')
        
        if not p_values:
            return {'error': 'No p-values found for correction'}
        
        p_values = np.array(p_values)
        
        # Apply different correction methods
        for method in self.multiple_testing_methods:
            try:
                rejected, p_corrected, _, _ = multipletests(p_values, alpha=self.significance_level, method=method)
                
                corrections[method] = {
                    'corrected_p_values': dict(zip(test_names, p_corrected)),
                    'rejected': dict(zip(test_names, rejected)),
                    'any_significant': np.any(rejected),
                    'num_significant': np.sum(rejected)
                }
            except Exception as e:
                corrections[method] = {'error': str(e)}
        
        return corrections
        
    def _block_bootstrap(self, returns: np.ndarray) -> Dict[str, Any]:
        """
        Block bootstrap for time series data to account for autocorrelation
        """
        n = len(returns)
        block_results = []
        
        for _ in range(self.n_bootstrap):
            # Generate block bootstrap sample
            bootstrap_sample = []
            current_pos = 0
            
            while current_pos < n:
                # Random starting position for block
                start_idx = np.random.randint(0, max(1, n - self.block_size + 1))
                block = returns[start_idx:start_idx + self.block_size]
                
                # Add block to sample
                remaining = n - current_pos
                if remaining >= len(block):
                    bootstrap_sample.extend(block)
                else:
                    bootstrap_sample.extend(block[:remaining])
                
                current_pos += len(block)
            
            # Calculate statistics
            bootstrap_sample = np.array(bootstrap_sample[:n])
            block_results.append({
                'mean': np.mean(bootstrap_sample),
                'std': np.std(bootstrap_sample),
                'sharpe': np.mean(bootstrap_sample) / np.std(bootstrap_sample) * np.sqrt(252 * 78) if np.std(bootstrap_sample) > 0 else 0
            })
        
        # Aggregate results
        means = [r['mean'] for r in block_results]
        stds = [r['std'] for r in block_results]
        sharpes = [r['sharpe'] for r in block_results]
        
        return {
            'mean_alpha': {
                'estimate': np.mean(means),
                'std_error': np.std(means),
                'confidence_interval': np.percentile(means, [2.5, 97.5])
            },
            'volatility': {
                'estimate': np.mean(stds),
                'std_error': np.std(stds),
                'confidence_interval': np.percentile(stds, [2.5, 97.5])
            },
            'sharpe_ratio': {
                'estimate': np.mean(sharpes),
                'std_error': np.std(sharpes),
                'confidence_interval': np.percentile(sharpes, [2.5, 97.5])
            }
        }
        
    def _permutation_tests(
        self,
        marl_result: BacktestResult,
        baseline_result: BacktestResult
    ) -> Dict[str, Any]:
        """
        Permutation tests for robust significance testing
        """
        permutation_results = {}
        
        # Prepare data
        min_len = min(len(marl_result.returns), len(baseline_result.returns))
        marl_returns = marl_result.returns[:min_len]
        baseline_returns = baseline_result.returns[:min_len]
        
        # 1. Permutation test for mean difference
        observed_diff = np.mean(marl_returns) - np.mean(baseline_returns)
        combined_returns = np.concatenate([marl_returns, baseline_returns])
        
        perm_diffs = []
        for _ in range(self.n_bootstrap):
            # Randomly permute the combined data
            permuted = np.random.permutation(combined_returns)
            perm_marl = permuted[:len(marl_returns)]
            perm_baseline = permuted[len(marl_returns):]
            
            perm_diff = np.mean(perm_marl) - np.mean(perm_baseline)
            perm_diffs.append(perm_diff)
        
        # Calculate p-value
        p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
        
        permutation_results['mean_difference'] = {
            'observed_statistic': observed_diff,
            'p_value': p_value,
            'significant': p_value < self.significance_level,
            'null_distribution_mean': np.mean(perm_diffs),
            'null_distribution_std': np.std(perm_diffs)
        }
        
        # 2. Permutation test for Sharpe ratio difference
        observed_sharpe_marl = np.mean(marl_returns) / np.std(marl_returns) * np.sqrt(252 * 78) if np.std(marl_returns) > 0 else 0
        observed_sharpe_baseline = np.mean(baseline_returns) / np.std(baseline_returns) * np.sqrt(252 * 78) if np.std(baseline_returns) > 0 else 0
        observed_sharpe_diff = observed_sharpe_marl - observed_sharpe_baseline
        
        perm_sharpe_diffs = []
        for _ in range(self.n_bootstrap):
            permuted = np.random.permutation(combined_returns)
            perm_marl = permuted[:len(marl_returns)]
            perm_baseline = permuted[len(marl_returns):]
            
            perm_sharpe_marl = np.mean(perm_marl) / np.std(perm_marl) * np.sqrt(252 * 78) if np.std(perm_marl) > 0 else 0
            perm_sharpe_baseline = np.mean(perm_baseline) / np.std(perm_baseline) * np.sqrt(252 * 78) if np.std(perm_baseline) > 0 else 0
            perm_sharpe_diff = perm_sharpe_marl - perm_sharpe_baseline
            
            perm_sharpe_diffs.append(perm_sharpe_diff)
        
        p_value_sharpe = np.mean(np.abs(perm_sharpe_diffs) >= np.abs(observed_sharpe_diff))
        
        permutation_results['sharpe_difference'] = {
            'observed_statistic': observed_sharpe_diff,
            'p_value': p_value_sharpe,
            'significant': p_value_sharpe < self.significance_level,
            'null_distribution_mean': np.mean(perm_sharpe_diffs),
            'null_distribution_std': np.std(perm_sharpe_diffs)
        }
        
        return permutation_results
        
    def _hac_standard_errors(
        self,
        marl_result: BacktestResult,
        baseline_result: BacktestResult
    ) -> Dict[str, Any]:
        """
        Newey-West HAC (Heteroskedasticity and Autocorrelation Consistent) standard errors
        """
        hac_results = {}
        
        # Prepare data
        min_len = min(len(marl_result.returns), len(baseline_result.returns))
        marl_returns = marl_result.returns[:min_len]
        baseline_returns = baseline_result.returns[:min_len]
        excess_returns = marl_returns - baseline_returns
        
        # Create design matrix (constant term for testing mean)
        n = len(excess_returns)
        X = np.ones((n, 1))
        y = excess_returns.reshape(-1, 1)
        
        # OLS regression
        try:
            model = OLS(y, X)
            results = model.fit()
            
            # Calculate HAC standard errors
            hac_cov = cov_hac(results, nlags=min(int(4 * (n/100)**(2/9)), n-1))
            hac_se = np.sqrt(np.diag(hac_cov))
            
            # Calculate t-statistic and p-value with HAC standard errors
            alpha_estimate = results.params[0]
            hac_t_stat = alpha_estimate / hac_se[0]
            hac_p_value = 2 * (1 - stats.t.cdf(np.abs(hac_t_stat), df=n-1))
            
            hac_results['alpha_test'] = {
                'estimate': alpha_estimate,
                'standard_error': hac_se[0],
                'ols_standard_error': results.bse[0],
                't_statistic': hac_t_stat,
                'p_value': hac_p_value,
                'significant': hac_p_value < self.significance_level,
                'confidence_interval': [
                    alpha_estimate - 1.96 * hac_se[0],
                    alpha_estimate + 1.96 * hac_se[0]
                ]
            }
            
            # Test for autocorrelation
            dw_stat = durbin_watson(results.resid)
            lb_stat, lb_p_value = acorr_ljungbox(results.resid, lags=10, return_df=False)
            
            hac_results['diagnostics'] = {
                'durbin_watson': dw_stat,
                'ljung_box_statistic': lb_stat,
                'ljung_box_p_value': lb_p_value,
                'autocorrelation_detected': lb_p_value < 0.05
            }
            
            # Augmented Dickey-Fuller test for stationarity
            adf_stat, adf_p_value, _, _, adf_critical, _ = adfuller(excess_returns)
            
            hac_results['stationarity'] = {
                'adf_statistic': adf_stat,
                'adf_p_value': adf_p_value,
                'critical_values': adf_critical,
                'is_stationary': adf_p_value < 0.05
            }
            
        except Exception as e:
            hac_results['error'] = f\"HAC analysis failed: {str(e)}\"
            
        return hac_results
        
    def _calculate_risk_adjusted_alpha(
        self,
        marl_result: BacktestResult,
        baseline_result: BacktestResult
    ) -> Dict[str, float]:
        """Calculate various measures of risk-adjusted alpha"""
        alpha_metrics = {}
        
        # 1. Jensen's Alpha (simplified - assuming baseline is market)
        marl_mean = np.mean(marl_result.returns)
        base_mean = np.mean(baseline_result.returns)
        marl_beta = np.cov(marl_result.returns[:len(baseline_result.returns)], 
                          baseline_result.returns)[0, 1] / np.var(baseline_result.returns)
        
        jensens_alpha = marl_mean - base_mean * marl_beta
        alpha_metrics['jensens_alpha'] = jensens_alpha
        alpha_metrics['beta'] = marl_beta
        
        # 2. Information Ratio
        excess_returns = marl_result.returns[:len(baseline_result.returns)] - baseline_result.returns
        tracking_error = np.std(excess_returns)
        
        if tracking_error > 0:
            info_ratio = np.mean(excess_returns) / tracking_error * np.sqrt(252 * 78)
        else:
            info_ratio = 0.0
            
        alpha_metrics['information_ratio'] = info_ratio
        alpha_metrics['tracking_error'] = tracking_error
        
        # 3. Alpha t-statistic
        if tracking_error > 0:
            alpha_t_stat = np.mean(excess_returns) / (tracking_error / np.sqrt(len(excess_returns)))
        else:
            alpha_t_stat = 0.0
            
        alpha_metrics['alpha_t_statistic'] = alpha_t_stat
        
        # 4. Downside capture ratio
        baseline_negative = baseline_result.returns[baseline_result.returns < 0]
        if len(baseline_negative) > 0:
            marl_negative = marl_result.returns[:len(baseline_result.returns)][baseline_result.returns < 0]
            downside_capture = np.mean(marl_negative) / np.mean(baseline_negative)
        else:
            downside_capture = 0.0
            
        alpha_metrics['downside_capture'] = downside_capture
        
        # 5. Upside capture ratio
        baseline_positive = baseline_result.returns[baseline_result.returns > 0]
        if len(baseline_positive) > 0:
            marl_positive = marl_result.returns[:len(baseline_result.returns)][baseline_result.returns > 0]
            upside_capture = np.mean(marl_positive) / np.mean(baseline_positive)
        else:
            upside_capture = 0.0
            
        alpha_metrics['upside_capture'] = upside_capture
        
        return alpha_metrics
        
    def _analyze_consistency(
        self,
        marl_result: BacktestResult,
        baseline_result: BacktestResult
    ) -> Dict[str, Any]:
        """Analyze consistency of alpha generation"""
        consistency = {}
        
        # 1. Rolling window analysis
        window_size = 20  # 20 periods
        
        marl_rolling_returns = pd.Series(marl_result.returns).rolling(window_size).mean()
        base_rolling_returns = pd.Series(baseline_result.returns[:len(marl_result.returns)]).rolling(window_size).mean()
        
        # Count periods where MARL outperforms
        outperformance = (marl_rolling_returns > base_rolling_returns).sum()
        total_windows = len(marl_rolling_returns.dropna())
        
        consistency['rolling_win_rate'] = outperformance / total_windows if total_windows > 0 else 0
        
        # 2. Monthly performance (assuming ~1560 periods per month)
        periods_per_month = 1560
        
        monthly_marl = self._calculate_period_returns(marl_result.returns, periods_per_month)
        monthly_base = self._calculate_period_returns(baseline_result.returns, periods_per_month)
        
        monthly_outperform = sum(m > b for m, b in zip(monthly_marl, monthly_base))
        consistency['monthly_win_rate'] = monthly_outperform / len(monthly_marl) if monthly_marl else 0
        
        # 3. Maximum consecutive underperformance
        excess = marl_result.returns[:len(baseline_result.returns)] - baseline_result.returns
        consecutive_negative = self._max_consecutive_negative(excess)
        consistency['max_consecutive_underperform'] = consecutive_negative
        
        # 4. Stability of Sharpe ratio
        rolling_sharpe_marl = self._rolling_sharpe(marl_result.returns, window_size)
        rolling_sharpe_base = self._rolling_sharpe(baseline_result.returns, window_size)
        
        consistency['sharpe_stability_marl'] = np.std(rolling_sharpe_marl)
        consistency['sharpe_stability_baseline'] = np.std(rolling_sharpe_base)
        
        return consistency
        
    def _generate_verdict(self, validation: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall verdict on alpha generation"""
        verdict = {
            'generates_alpha': False,
            'confidence': 'low',
            'recommendation': '',
            'key_strengths': [],
            'key_weaknesses': []
        }
        
        # Check performance improvements
        perf = validation['performance']
        positive_metrics = sum(1 for k, v in perf.items() 
                             if k.endswith('_improvement') and v > 0)
        
        # Check statistical significance
        stats = validation['statistical_tests']
        significant_tests = sum(1 for test in stats.values() 
                              if isinstance(test, dict) and test.get('significant', False))
        
        # Check advanced bootstrap significance
        advanced_bootstrap = validation.get('advanced_bootstrap', {})
        bca_significant = advanced_bootstrap.get('bca_alpha', {}).get('significant', False)
        studentized_significant = advanced_bootstrap.get('studentized_alpha', {}).get('significant', False)
        
        # Check multiple testing corrections
        multiple_testing = validation.get('multiple_testing', {})
        bonferroni_significant = multiple_testing.get('bonferroni', {}).get('any_significant', False)
        fdr_significant = multiple_testing.get('fdr_bh', {}).get('any_significant', False)
        
        # Check power analysis
        power_analysis = validation.get('power_analysis', {})
        adequate_power = power_analysis.get('current_power', 0) > 0.8
        large_effect_size = abs(power_analysis.get('effect_size', 0)) > 0.5
        
        # Check risk-adjusted alpha
        risk_adj = validation['risk_adjusted_alpha']
        positive_alpha = risk_adj['jensens_alpha'] > 0
        good_info_ratio = risk_adj['information_ratio'] > 0.5
        
        # Check consistency
        consistency = validation['consistency']
        good_consistency = consistency['rolling_win_rate'] > 0.55
        
        # Advanced significance indicators
        robust_significance = sum([
            bca_significant,
            studentized_significant,
            bonferroni_significant,
            fdr_significant
        ])
        
        # Determine verdict with enhanced criteria
        if (positive_metrics >= 4 and significant_tests >= 2 and 
            robust_significance >= 2 and positive_alpha and good_consistency and adequate_power):
            verdict['generates_alpha'] = True
            verdict['confidence'] = 'very_high'
            verdict['recommendation'] = 'MARL system shows very strong evidence of alpha generation with robust statistical support'
        elif (positive_metrics >= 4 and significant_tests >= 2 and 
              robust_significance >= 1 and positive_alpha and good_consistency):
            verdict['generates_alpha'] = True
            verdict['confidence'] = 'high'
            verdict['recommendation'] = 'MARL system shows strong evidence of alpha generation'
        elif (positive_metrics >= 3 and significant_tests >= 1 and 
              (positive_alpha or good_info_ratio) and robust_significance >= 1):
            verdict['generates_alpha'] = True
            verdict['confidence'] = 'medium'
            verdict['recommendation'] = 'MARL system shows moderate evidence of alpha generation'
        else:
            verdict['generates_alpha'] = False
            verdict['confidence'] = 'low'
            verdict['recommendation'] = 'MARL system does not show sufficient evidence of alpha generation'
            
        # Identify strengths
        if perf['sharpe_ratio_improvement'] > 0.2:
            verdict['key_strengths'].append('Significant Sharpe ratio improvement')
        if risk_adj['information_ratio'] > 1.0:
            verdict['key_strengths'].append('High information ratio')
        if consistency['rolling_win_rate'] > 0.6:
            verdict['key_strengths'].append('Consistent outperformance')
        if bca_significant:
            verdict['key_strengths'].append('Robust BCa bootstrap significance')
        if adequate_power:
            verdict['key_strengths'].append('Adequate statistical power')
        if large_effect_size:
            verdict['key_strengths'].append('Large effect size')
            
        # Identify weaknesses
        if perf['max_drawdown_improvement'] < 0:
            verdict['key_weaknesses'].append('Higher drawdown than baseline')
        if consistency['max_consecutive_underperform'] > 50:
            verdict['key_weaknesses'].append('Long periods of underperformance')
        if not stats['ttest']['significant']:
            verdict['key_weaknesses'].append('Returns not statistically different')
        if not adequate_power:
            verdict['key_weaknesses'].append('Insufficient statistical power')
        if not bonferroni_significant:
            verdict['key_weaknesses'].append('Not significant after Bonferroni correction')
            
        return verdict
        
    def _calculate_period_returns(
        self,
        returns: np.ndarray,
        period_size: int
    ) -> List[float]:
        """Calculate returns for fixed periods"""
        period_returns = []
        
        for i in range(0, len(returns), period_size):
            period_ret = returns[i:i+period_size]
            if len(period_ret) > 0:
                # Compound returns
                compound_return = np.prod(1 + period_ret) - 1
                period_returns.append(compound_return)
                
        return period_returns
        
    def _max_consecutive_negative(self, returns: np.ndarray) -> int:
        """Find maximum consecutive negative returns"""
        max_consecutive = 0
        current_consecutive = 0
        
        for ret in returns:
            if ret < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
                
        return max_consecutive
        
    def _rolling_sharpe(
        self,
        returns: np.ndarray,
        window: int
    ) -> List[float]:
        """Calculate rolling Sharpe ratio"""
        rolling_sharpe = []
        
        for i in range(window, len(returns)):
            window_returns = returns[i-window:i]
            if np.std(window_returns) > 0:
                sharpe = np.mean(window_returns) / np.std(window_returns) * np.sqrt(252 * 78)
            else:
                sharpe = 0.0
            rolling_sharpe.append(sharpe)
            
        return rolling_sharpe
        
    def generate_report(self, save_path: Optional[str] = None) -> str:
        """Generate comprehensive alpha validation report"""
        if not self.results:
            return "No validation results available"
            
        report = []
        report.append("=" * 60)
        report.append("STRATEGIC MARL ALPHA VALIDATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Performance Summary
        report.append("PERFORMANCE COMPARISON")
        report.append("-" * 40)
        perf = self.results['performance']
        for metric in ['total_return', 'sharpe_ratio', 'max_drawdown', 'calmar_ratio']:
            marl_val = perf.get(f'{metric}_marl', 0)
            base_val = perf.get(f'{metric}_baseline', 0)
            improvement = perf.get(f'{metric}_improvement', 0)
            
            report.append(f"{metric.replace('_', ' ').title()}:")
            report.append(f"  MARL: {marl_val:.3f}")
            report.append(f"  Baseline: {base_val:.3f}")
            report.append(f"  Improvement: {improvement:.1%}")
            report.append("")
            
        # Statistical Tests
        report.append("STATISTICAL SIGNIFICANCE")
        report.append("-" * 40)
        stats = self.results['statistical_tests']
        
        report.append(f"T-test p-value: {stats['ttest']['pvalue']:.4f}")
        report.append(f"Significant at {self.significance_level} level: {stats['ttest']['significant']}")
        report.append(f"Mean excess return: {stats['paired_ttest']['mean_excess']:.4f}")
        report.append("")
        
        # Advanced Bootstrap Results
        if 'advanced_bootstrap' in self.results:
            report.append("ADVANCED BOOTSTRAP ANALYSIS")
            report.append("-" * 40)
            adv_bootstrap = self.results['advanced_bootstrap']
            
            if 'bca_alpha' in adv_bootstrap:
                bca = adv_bootstrap['bca_alpha']
                report.append(f"BCa Alpha CI: [{bca['confidence_interval'][0]:.4f}, {bca['confidence_interval'][1]:.4f}]")
                report.append(f"BCa Alpha Significant: {bca['significant']}")
                
            if 'studentized_alpha' in adv_bootstrap:
                stud = adv_bootstrap['studentized_alpha']
                report.append(f"Studentized Alpha CI: [{stud['confidence_interval'][0]:.4f}, {stud['confidence_interval'][1]:.4f}]")
                report.append(f"Studentized Alpha Significant: {stud['significant']}")
            report.append("")
            
        # Multiple Testing Corrections
        if 'multiple_testing' in self.results:
            report.append("MULTIPLE TESTING CORRECTIONS")
            report.append("-" * 40)
            mult_test = self.results['multiple_testing']
            
            for method, results in mult_test.items():
                if 'error' not in results:
                    report.append(f"{method.upper()}: {results['num_significant']}/{len(results['rejected'])} significant")
            report.append("")
            
        # Power Analysis
        if 'power_analysis' in self.results:
            report.append("POWER ANALYSIS")
            report.append("-" * 40)
            power = self.results['power_analysis']
            
            report.append(f"Effect Size (Cohen's d): {power['effect_size']:.3f}")
            report.append(f"Current Power: {power['current_power']:.3f}")
            report.append(f"Required N for 80% power: {power['required_sample_sizes']['power_0.8']}")
            report.append(f"Min Detectable Effect Size: {power['min_detectable_effect_size']:.3f}")
            report.append("")
        
        # Risk-Adjusted Alpha
        report.append("RISK-ADJUSTED ALPHA")
        report.append("-" * 40)
        risk_adj = self.results['risk_adjusted_alpha']
        
        report.append(f"Jensen's Alpha: {risk_adj['jensens_alpha']:.4f}")
        report.append(f"Information Ratio: {risk_adj['information_ratio']:.3f}")
        report.append(f"Beta vs Baseline: {risk_adj['beta']:.3f}")
        report.append(f"Downside Capture: {risk_adj['downside_capture']:.3f}")
        report.append(f"Upside Capture: {risk_adj['upside_capture']:.3f}")
        report.append("")
        
        # Consistency
        report.append("CONSISTENCY ANALYSIS")
        report.append("-" * 40)
        consistency = self.results['consistency']
        
        report.append(f"Rolling Win Rate: {consistency['rolling_win_rate']:.1%}")
        report.append(f"Monthly Win Rate: {consistency['monthly_win_rate']:.1%}")
        report.append(f"Max Consecutive Underperformance: {consistency['max_consecutive_underperform']} periods")
        report.append("")
        
        # Final Verdict
        report.append("FINAL VERDICT")
        report.append("-" * 40)
        verdict = self.results['verdict']
        
        report.append(f"Generates Alpha: {verdict['generates_alpha']}")
        report.append(f"Confidence Level: {verdict['confidence'].upper()}")
        report.append(f"Recommendation: {verdict['recommendation']}")
        
        if verdict['key_strengths']:
            report.append("\nKey Strengths:")
            for strength in verdict['key_strengths']:
                report.append(f"  • {strength}")
                
        if verdict['key_weaknesses']:
            report.append("\nKey Weaknesses:")
            for weakness in verdict['key_weaknesses']:
                report.append(f"  • {weakness}")
                
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
                
        return report_text


def main():
    """Example validation run"""
    logging.basicConfig(level=logging.INFO)
    
    # Run backtests
    runner = BacktestRunner()
    config = BacktestConfig(episodes=100, random_seed=42)
    
    # Get baseline result
    baseline_results = runner.run_baseline_agents(config)
    baseline_result = baseline_results['enhanced_rule']  # Use best baseline
    
    # For demonstration, create a mock "improved" MARL result
    # In production, this would load actual trained model
    mock_marl_result = BacktestResult(
        agent_name="marl",
        equity_curve=baseline_result.equity_curve * 1.15,  # 15% better
        returns=baseline_result.returns * 1.1 + np.random.normal(0, 0.0001, len(baseline_result.returns)),
        positions=baseline_result.positions,
        actions=baseline_result.actions,
        metrics=None,  # Will be recalculated
        execution_time=baseline_result.execution_time,
        config=config
    )
    
    # Recalculate metrics
    from .metrics import calculate_all_metrics
    mock_marl_result.metrics = calculate_all_metrics(
        mock_marl_result.equity_curve,
        mock_marl_result.returns
    )
    
    # Validate alpha
    validator = AlphaValidator()
    validation = validator.validate_alpha(mock_marl_result, baseline_result)
    
    # Generate report
    report = validator.generate_report("alpha_validation_report.txt")
    print(report)
    

if __name__ == "__main__":
    main()