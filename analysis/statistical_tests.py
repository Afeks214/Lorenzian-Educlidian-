"""
Advanced Statistical Tests for Alpha Validation

This module provides comprehensive statistical testing methods including
advanced bootstrap techniques, Bayesian hypothesis testing, and robust
statistical inference methods.
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from dataclasses import dataclass
from scipy import stats
from scipy.optimize import minimize_scalar, brentq
from scipy.special import beta, gamma, digamma, polygamma
from sklearn.utils import resample
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.weightstats import ttest_ind
from statsmodels.stats.contingency_tables import mcnemar
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Statistical test result"""
    test_name: str
    statistic: float
    p_value: float
    critical_value: Optional[float]
    confidence_interval: Optional[Tuple[float, float]]
    effect_size: Optional[float]
    power: Optional[float]
    reject_null: bool
    interpretation: str
    sample_size: int
    additional_stats: Dict[str, Any]


@dataclass
class BootstrapResult:
    """Bootstrap test result"""
    original_statistic: float
    bootstrap_statistics: np.ndarray
    confidence_interval: Tuple[float, float]
    p_value: float
    bias: float
    standard_error: float
    method: str
    n_bootstrap: int


@dataclass
class BayesianResult:
    """Bayesian test result"""
    bayes_factor: float
    posterior_probability: float
    prior_probability: float
    evidence: float
    credible_interval: Tuple[float, float]
    interpretation: str
    method: str


class AdvancedBootstrap:
    """
    Advanced bootstrap methods for statistical inference
    """
    
    def __init__(self, n_bootstrap: int = 1000, confidence_level: float = 0.95):
        """
        Initialize bootstrap analyzer
        
        Args:
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level for intervals
        """
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        
    def bias_corrected_accelerated(
        self,
        data: np.ndarray,
        statistic: Callable[[np.ndarray], float],
        original_stat: Optional[float] = None
    ) -> BootstrapResult:
        """
        Bias-Corrected and Accelerated (BCa) bootstrap
        
        Args:
            data: Input data
            statistic: Function to compute statistic
            original_stat: Original statistic value
            
        Returns:
            BCa bootstrap result
        """
        n = len(data)
        
        if original_stat is None:
            original_stat = statistic(data)
            
        # Bootstrap samples
        bootstrap_stats = []
        for _ in range(self.n_bootstrap):
            bootstrap_sample = resample(data, n_samples=n)
            bootstrap_stats.append(statistic(bootstrap_sample))
        
        bootstrap_stats = np.array(bootstrap_stats)
        
        # Bias correction
        bias_correction = stats.norm.ppf(np.mean(bootstrap_stats < original_stat))
        
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
        z_alpha = stats.norm.ppf(self.alpha / 2)
        z_1_alpha = stats.norm.ppf(1 - self.alpha / 2)
        
        # BCa adjusted percentiles
        alpha_1 = stats.norm.cdf(bias_correction + (bias_correction + z_alpha) / 
                                (1 - acceleration * (bias_correction + z_alpha)))
        alpha_2 = stats.norm.cdf(bias_correction + (bias_correction + z_1_alpha) / 
                                (1 - acceleration * (bias_correction + z_1_alpha)))
        
        # Ensure percentiles are within bounds
        alpha_1 = max(0, min(1, alpha_1))
        alpha_2 = max(0, min(1, alpha_2))
        
        # Calculate confidence interval
        lower = np.percentile(bootstrap_stats, alpha_1 * 100)
        upper = np.percentile(bootstrap_stats, alpha_2 * 100)
        
        # Calculate p-value (two-tailed)
        p_value = 2 * min(np.mean(bootstrap_stats >= original_stat), 
                         np.mean(bootstrap_stats <= original_stat))
        
        return BootstrapResult(
            original_statistic=original_stat,
            bootstrap_statistics=bootstrap_stats,
            confidence_interval=(lower, upper),
            p_value=p_value,
            bias=np.mean(bootstrap_stats) - original_stat,
            standard_error=np.std(bootstrap_stats),
            method='BCa',
            n_bootstrap=self.n_bootstrap
        )
        
    def studentized_bootstrap(
        self,
        data: np.ndarray,
        statistic: Callable[[np.ndarray], float],
        variance_func: Optional[Callable[[np.ndarray], float]] = None
    ) -> BootstrapResult:
        """
        Studentized (pivot) bootstrap
        
        Args:
            data: Input data
            statistic: Function to compute statistic
            variance_func: Function to compute variance of statistic
            
        Returns:
            Studentized bootstrap result
        """
        n = len(data)
        original_stat = statistic(data)
        
        if variance_func is None:
            # Default variance function for mean
            variance_func = lambda x: np.var(x, ddof=1) / len(x)
        
        # First-level bootstrap
        bootstrap_stats = []
        studentized_stats = []
        
        for _ in range(self.n_bootstrap):
            bootstrap_sample = resample(data, n_samples=n)
            bootstrap_stat = statistic(bootstrap_sample)
            bootstrap_stats.append(bootstrap_stat)
            
            # Second-level bootstrap for variance estimation
            second_level_stats = []
            for _ in range(min(100, self.n_bootstrap // 10)):  # Reduced for speed
                second_bootstrap = resample(bootstrap_sample, n_samples=n)
                second_level_stats.append(statistic(second_bootstrap))
            
            bootstrap_variance = np.var(second_level_stats, ddof=1)
            
            if bootstrap_variance > 0:
                studentized_stats.append((bootstrap_stat - original_stat) / np.sqrt(bootstrap_variance))
            else:
                studentized_stats.append(0)
        
        studentized_stats = np.array(studentized_stats)
        bootstrap_stats = np.array(bootstrap_stats)
        
        # Calculate original variance
        original_variance = variance_func(data)
        
        # Get critical values
        t_alpha = np.percentile(studentized_stats, (self.alpha / 2) * 100)
        t_1_alpha = np.percentile(studentized_stats, (1 - self.alpha / 2) * 100)
        
        # Construct confidence interval
        lower = original_stat - t_1_alpha * np.sqrt(original_variance)
        upper = original_stat - t_alpha * np.sqrt(original_variance)
        
        # Calculate p-value
        p_value = 2 * min(np.mean(studentized_stats >= 0), 
                         np.mean(studentized_stats <= 0))
        
        return BootstrapResult(
            original_statistic=original_stat,
            bootstrap_statistics=bootstrap_stats,
            confidence_interval=(lower, upper),
            p_value=p_value,
            bias=np.mean(bootstrap_stats) - original_stat,
            standard_error=np.std(bootstrap_stats),
            method='Studentized',
            n_bootstrap=self.n_bootstrap
        )
        
    def smooth_bootstrap(
        self,
        data: np.ndarray,
        statistic: Callable[[np.ndarray], float],
        bandwidth: Optional[float] = None
    ) -> BootstrapResult:
        """
        Smooth bootstrap using kernel density estimation
        
        Args:
            data: Input data
            statistic: Function to compute statistic
            bandwidth: Kernel bandwidth (auto-selected if None)
            
        Returns:
            Smooth bootstrap result
        """
        n = len(data)
        original_stat = statistic(data)
        
        # Estimate bandwidth using Silverman's rule
        if bandwidth is None:
            bandwidth = 1.06 * np.std(data) * n ** (-1/5)
            
        # Bootstrap samples with kernel smoothing
        bootstrap_stats = []
        for _ in range(self.n_bootstrap):
            # Resample with replacement
            bootstrap_indices = np.random.choice(n, size=n, replace=True)
            bootstrap_sample = data[bootstrap_indices]
            
            # Add kernel noise
            noise = np.random.normal(0, bandwidth, n)
            smooth_sample = bootstrap_sample + noise
            
            bootstrap_stats.append(statistic(smooth_sample))
        
        bootstrap_stats = np.array(bootstrap_stats)
        
        # Calculate confidence interval
        lower = np.percentile(bootstrap_stats, (self.alpha / 2) * 100)
        upper = np.percentile(bootstrap_stats, (1 - self.alpha / 2) * 100)
        
        # Calculate p-value
        p_value = 2 * min(np.mean(bootstrap_stats >= original_stat), 
                         np.mean(bootstrap_stats <= original_stat))
        
        return BootstrapResult(
            original_statistic=original_stat,
            bootstrap_statistics=bootstrap_stats,
            confidence_interval=(lower, upper),
            p_value=p_value,
            bias=np.mean(bootstrap_stats) - original_stat,
            standard_error=np.std(bootstrap_stats),
            method='Smooth',
            n_bootstrap=self.n_bootstrap
        )
        
    def block_bootstrap(
        self,
        data: np.ndarray,
        statistic: Callable[[np.ndarray], float],
        block_size: Optional[int] = None
    ) -> BootstrapResult:
        """
        Block bootstrap for time series data
        
        Args:
            data: Time series data
            statistic: Function to compute statistic
            block_size: Block size (auto-selected if None)
            
        Returns:
            Block bootstrap result
        """
        n = len(data)
        original_stat = statistic(data)
        
        # Choose block size using optimal rate
        if block_size is None:
            block_size = int(n ** (1/3))
            
        bootstrap_stats = []
        for _ in range(self.n_bootstrap):
            # Generate block bootstrap sample
            bootstrap_sample = []
            current_pos = 0
            
            while current_pos < n:
                # Random starting position for block
                start_idx = np.random.randint(0, max(1, n - block_size + 1))
                block = data[start_idx:start_idx + block_size]
                
                # Add block to sample
                remaining = n - current_pos
                if remaining >= len(block):
                    bootstrap_sample.extend(block)
                else:
                    bootstrap_sample.extend(block[:remaining])
                
                current_pos += len(block)
            
            bootstrap_sample = np.array(bootstrap_sample[:n])
            bootstrap_stats.append(statistic(bootstrap_sample))
        
        bootstrap_stats = np.array(bootstrap_stats)
        
        # Calculate confidence interval
        lower = np.percentile(bootstrap_stats, (self.alpha / 2) * 100)
        upper = np.percentile(bootstrap_stats, (1 - self.alpha / 2) * 100)
        
        # Calculate p-value
        p_value = 2 * min(np.mean(bootstrap_stats >= original_stat), 
                         np.mean(bootstrap_stats <= original_stat))
        
        return BootstrapResult(
            original_statistic=original_stat,
            bootstrap_statistics=bootstrap_stats,
            confidence_interval=(lower, upper),
            p_value=p_value,
            bias=np.mean(bootstrap_stats) - original_stat,
            standard_error=np.std(bootstrap_stats),
            method='Block',
            n_bootstrap=self.n_bootstrap
        )


class BayesianTesting:
    """
    Bayesian hypothesis testing methods
    """
    
    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        """
        Initialize Bayesian testing
        
        Args:
            prior_alpha: Prior alpha parameter
            prior_beta: Prior beta parameter
        """
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        
    def bayesian_t_test(
        self,
        data: np.ndarray,
        mu_0: float = 0,
        prior_mean: float = 0,
        prior_precision: float = 1.0,
        prior_df: float = 1.0,
        prior_scale: float = 1.0
    ) -> BayesianResult:
        """
        Bayesian t-test using conjugate priors
        
        Args:
            data: Sample data
            mu_0: Null hypothesis mean
            prior_mean: Prior mean
            prior_precision: Prior precision
            prior_df: Prior degrees of freedom
            prior_scale: Prior scale parameter
            
        Returns:
            Bayesian test result
        """
        n = len(data)
        sample_mean = np.mean(data)
        sample_var = np.var(data, ddof=1)
        
        # Posterior parameters
        posterior_precision = prior_precision + n
        posterior_mean = (prior_precision * prior_mean + n * sample_mean) / posterior_precision
        posterior_df = prior_df + n
        posterior_scale = prior_scale + 0.5 * (n - 1) * sample_var + \
                         0.5 * (prior_precision * n / posterior_precision) * (sample_mean - prior_mean) ** 2
        
        # Marginal likelihood under null and alternative
        # Simplified calculation using normal approximation
        posterior_var = posterior_scale / posterior_df
        
        # Bayes factor approximation
        # BF = p(data | H1) / p(data | H0)
        log_bf = -0.5 * n * np.log(2 * np.pi * sample_var) - \
                 0.5 * np.sum((data - posterior_mean) ** 2) / sample_var + \
                 0.5 * n * np.log(2 * np.pi * posterior_var) + \
                 0.5 * np.sum((data - mu_0) ** 2) / posterior_var
        
        bayes_factor = np.exp(log_bf)
        
        # Posterior probability assuming equal prior probabilities
        posterior_prob = bayes_factor / (1 + bayes_factor)
        
        # Credible interval
        from scipy.stats import t as t_dist
        t_critical = t_dist.ppf(0.975, posterior_df)
        margin_error = t_critical * np.sqrt(posterior_var / n)
        credible_interval = (posterior_mean - margin_error, posterior_mean + margin_error)
        
        # Interpretation
        if bayes_factor > 10:
            interpretation = "Strong evidence for alternative hypothesis"
        elif bayes_factor > 3:
            interpretation = "Moderate evidence for alternative hypothesis"
        elif bayes_factor > 1:
            interpretation = "Weak evidence for alternative hypothesis"
        elif bayes_factor > 0.33:
            interpretation = "Weak evidence for null hypothesis"
        elif bayes_factor > 0.1:
            interpretation = "Moderate evidence for null hypothesis"
        else:
            interpretation = "Strong evidence for null hypothesis"
        
        return BayesianResult(
            bayes_factor=bayes_factor,
            posterior_probability=posterior_prob,
            prior_probability=0.5,
            evidence=log_bf,
            credible_interval=credible_interval,
            interpretation=interpretation,
            method='Conjugate Prior T-test'
        )
        
    def bayesian_proportion_test(
        self,
        successes: int,
        trials: int,
        p_0: float = 0.5
    ) -> BayesianResult:
        """
        Bayesian proportion test using Beta-Binomial conjugacy
        
        Args:
            successes: Number of successes
            trials: Number of trials
            p_0: Null hypothesis proportion
            
        Returns:
            Bayesian test result
        """
        # Posterior parameters (Beta distribution)
        posterior_alpha = self.prior_alpha + successes
        posterior_beta = self.prior_beta + trials - successes
        
        # Posterior mean and variance
        posterior_mean = posterior_alpha / (posterior_alpha + posterior_beta)
        posterior_var = (posterior_alpha * posterior_beta) / \
                       ((posterior_alpha + posterior_beta) ** 2 * (posterior_alpha + posterior_beta + 1))
        
        # Bayes factor calculation
        # Marginal likelihood under null
        from scipy.special import beta as beta_func
        
        marginal_null = beta_func(self.prior_alpha, self.prior_beta) * \
                       stats.binom.pmf(successes, trials, p_0)
        
        # Marginal likelihood under alternative (integrated over prior)
        marginal_alt = beta_func(posterior_alpha, posterior_beta) / \
                      beta_func(self.prior_alpha, self.prior_beta) * \
                      stats.binom.pmf(successes, trials, posterior_mean)
        
        bayes_factor = marginal_alt / marginal_null
        posterior_prob = bayes_factor / (1 + bayes_factor)
        
        # Credible interval
        from scipy.stats import beta as beta_dist
        credible_interval = (
            beta_dist.ppf(0.025, posterior_alpha, posterior_beta),
            beta_dist.ppf(0.975, posterior_alpha, posterior_beta)
        )
        
        # Interpretation
        if bayes_factor > 10:
            interpretation = "Strong evidence for alternative hypothesis"
        elif bayes_factor > 3:
            interpretation = "Moderate evidence for alternative hypothesis"
        elif bayes_factor > 1:
            interpretation = "Weak evidence for alternative hypothesis"
        elif bayes_factor > 0.33:
            interpretation = "Weak evidence for null hypothesis"
        elif bayes_factor > 0.1:
            interpretation = "Moderate evidence for null hypothesis"
        else:
            interpretation = "Strong evidence for null hypothesis"
        
        return BayesianResult(
            bayes_factor=bayes_factor,
            posterior_probability=posterior_prob,
            prior_probability=0.5,
            evidence=np.log(bayes_factor),
            credible_interval=credible_interval,
            interpretation=interpretation,
            method='Beta-Binomial'
        )


class RobustTesting:
    """
    Robust statistical testing methods
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize robust testing
        
        Args:
            significance_level: Significance level for tests
        """
        self.significance_level = significance_level
        
    def trimmed_mean_test(
        self,
        data: np.ndarray,
        trim_proportion: float = 0.1,
        mu_0: float = 0
    ) -> TestResult:
        """
        Test using trimmed mean for robustness to outliers
        
        Args:
            data: Sample data
            trim_proportion: Proportion to trim from each tail
            mu_0: Null hypothesis mean
            
        Returns:
            Test result
        """
        # Trimmed mean
        trimmed_data = stats.trim_mean(data, trim_proportion)
        
        # Winsorized variance for standard error
        winsorized_var = stats.mstats.winsorize(data, limits=trim_proportion).var(ddof=1)
        
        # Effective sample size
        n_effective = len(data) * (1 - 2 * trim_proportion)
        
        # Test statistic
        se = np.sqrt(winsorized_var / n_effective)
        t_stat = (trimmed_data - mu_0) / se
        
        # Degrees of freedom (approximate)
        df = n_effective - 1
        
        # p-value
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
        
        # Confidence interval
        t_critical = stats.t.ppf(1 - self.significance_level / 2, df)
        ci_lower = trimmed_data - t_critical * se
        ci_upper = trimmed_data + t_critical * se
        
        return TestResult(
            test_name='Trimmed Mean Test',
            statistic=t_stat,
            p_value=p_value,
            critical_value=t_critical,
            confidence_interval=(ci_lower, ci_upper),
            effect_size=(trimmed_data - mu_0) / np.sqrt(winsorized_var),
            power=None,
            reject_null=p_value < self.significance_level,
            interpretation=f"Trimmed mean ({trim_proportion*100:.0f}% trimmed) test",
            sample_size=len(data),
            additional_stats={
                'trimmed_mean': trimmed_data,
                'winsorized_variance': winsorized_var,
                'effective_n': n_effective
            }
        )
        
    def median_test(
        self,
        data: np.ndarray,
        median_0: float = 0
    ) -> TestResult:
        """
        Robust median test using sign test
        
        Args:
            data: Sample data
            median_0: Null hypothesis median
            
        Returns:
            Test result
        """
        # Sign test statistic
        differences = data - median_0
        positive_count = np.sum(differences > 0)
        negative_count = np.sum(differences < 0)
        n_nonzero = positive_count + negative_count
        
        # Test statistic (number of positive differences)
        test_stat = positive_count
        
        # p-value using binomial distribution
        p_value = 2 * min(
            stats.binom.cdf(test_stat, n_nonzero, 0.5),
            1 - stats.binom.cdf(test_stat - 1, n_nonzero, 0.5)
        )
        
        # Confidence interval for median (approximate)
        sample_median = np.median(data)
        
        # Bootstrap confidence interval for median
        bootstrap_medians = []
        for _ in range(1000):
            bootstrap_sample = resample(data, n_samples=len(data))
            bootstrap_medians.append(np.median(bootstrap_sample))
        
        ci_lower = np.percentile(bootstrap_medians, 2.5)
        ci_upper = np.percentile(bootstrap_medians, 97.5)
        
        return TestResult(
            test_name='Median Test (Sign Test)',
            statistic=test_stat,
            p_value=p_value,
            critical_value=None,
            confidence_interval=(ci_lower, ci_upper),
            effect_size=None,
            power=None,
            reject_null=p_value < self.significance_level,
            interpretation="Non-parametric median test",
            sample_size=len(data),
            additional_stats={
                'sample_median': sample_median,
                'positive_count': positive_count,
                'negative_count': negative_count,
                'ties': len(data) - n_nonzero
            }
        )
        
    def rank_sum_test(
        self,
        data1: np.ndarray,
        data2: np.ndarray
    ) -> TestResult:
        """
        Wilcoxon rank-sum test for two independent samples
        
        Args:
            data1: First sample
            data2: Second sample
            
        Returns:
            Test result
        """
        # Wilcoxon rank-sum test
        statistic, p_value = stats.ranksums(data1, data2)
        
        # Effect size (rank-biserial correlation)
        n1, n2 = len(data1), len(data2)
        U1, _ = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        r = 2 * U1 / (n1 * n2) - 1  # Rank-biserial correlation
        
        # Confidence interval (approximate)
        se = np.sqrt((n1 + n2 + 1) / (3 * n1 * n2))
        z_critical = stats.norm.ppf(1 - self.significance_level / 2)
        ci_lower = r - z_critical * se
        ci_upper = r + z_critical * se
        
        return TestResult(
            test_name='Wilcoxon Rank-Sum Test',
            statistic=statistic,
            p_value=p_value,
            critical_value=z_critical,
            confidence_interval=(ci_lower, ci_upper),
            effect_size=r,
            power=None,
            reject_null=p_value < self.significance_level,
            interpretation="Non-parametric test for difference in distributions",
            sample_size=n1 + n2,
            additional_stats={
                'n1': n1,
                'n2': n2,
                'U_statistic': U1,
                'rank_biserial_correlation': r
            }
        )


class StatisticalTestSuite:
    """
    Comprehensive statistical testing suite
    """
    
    def __init__(self, significance_level: float = 0.05, n_bootstrap: int = 1000):
        """
        Initialize test suite
        
        Args:
            significance_level: Significance level for tests
            n_bootstrap: Number of bootstrap samples
        """
        self.significance_level = significance_level
        self.n_bootstrap = n_bootstrap
        
        # Initialize component classes
        self.bootstrap = AdvancedBootstrap(n_bootstrap, 1 - significance_level)
        self.bayesian = BayesianTesting()
        self.robust = RobustTesting(significance_level)
        
    def comprehensive_alpha_test(
        self,
        portfolio_returns: np.ndarray,
        benchmark_returns: np.ndarray
    ) -> Dict[str, Any]:
        """
        Comprehensive alpha testing using multiple methods
        
        Args:
            portfolio_returns: Portfolio return series
            benchmark_returns: Benchmark return series
            
        Returns:
            Dictionary with all test results
        """
        results = {}
        
        # Ensure same length
        min_len = min(len(portfolio_returns), len(benchmark_returns))
        portfolio_returns = portfolio_returns[:min_len]
        benchmark_returns = benchmark_returns[:min_len]
        excess_returns = portfolio_returns - benchmark_returns
        
        # 1. Classical tests
        results['classical'] = self._classical_tests(excess_returns)
        
        # 2. Bootstrap tests
        results['bootstrap'] = self._bootstrap_tests(excess_returns)
        
        # 3. Bayesian tests
        results['bayesian'] = self._bayesian_tests(excess_returns)
        
        # 4. Robust tests
        results['robust'] = self._robust_tests(excess_returns)
        
        # 5. Permutation tests
        results['permutation'] = self._permutation_tests(portfolio_returns, benchmark_returns)
        
        # 6. Multiple testing corrections
        results['multiple_testing'] = self._multiple_testing_corrections(results)
        
        # 7. Power analysis
        results['power_analysis'] = self._power_analysis(excess_returns)
        
        # 8. Summary and recommendation
        results['summary'] = self._generate_summary(results)
        
        return results
        
    def _classical_tests(self, excess_returns: np.ndarray) -> Dict[str, TestResult]:
        """Run classical statistical tests"""
        tests = {}
        
        # One-sample t-test
        t_stat, p_value = stats.ttest_1samp(excess_returns, 0)
        n = len(excess_returns)
        df = n - 1
        t_critical = stats.t.ppf(1 - self.significance_level / 2, df)
        se = np.std(excess_returns, ddof=1) / np.sqrt(n)
        ci_lower = np.mean(excess_returns) - t_critical * se
        ci_upper = np.mean(excess_returns) + t_critical * se
        
        tests['t_test'] = TestResult(
            test_name='One-sample t-test',
            statistic=t_stat,
            p_value=p_value,
            critical_value=t_critical,
            confidence_interval=(ci_lower, ci_upper),
            effect_size=np.mean(excess_returns) / np.std(excess_returns, ddof=1),
            power=None,
            reject_null=p_value < self.significance_level,
            interpretation="Test if mean excess return is significantly different from zero",
            sample_size=n,
            additional_stats={
                'mean': np.mean(excess_returns),
                'std': np.std(excess_returns, ddof=1),
                'degrees_of_freedom': df
            }
        )
        
        # Wilcoxon signed-rank test (non-parametric)
        w_stat, w_p_value = stats.wilcoxon(excess_returns)
        
        tests['wilcoxon'] = TestResult(
            test_name='Wilcoxon signed-rank test',
            statistic=w_stat,
            p_value=w_p_value,
            critical_value=None,
            confidence_interval=None,
            effect_size=None,
            power=None,
            reject_null=w_p_value < self.significance_level,
            interpretation="Non-parametric test for median difference from zero",
            sample_size=n,
            additional_stats={
                'median': np.median(excess_returns)
            }
        )
        
        return tests
        
    def _bootstrap_tests(self, excess_returns: np.ndarray) -> Dict[str, BootstrapResult]:
        """Run bootstrap tests"""
        tests = {}
        
        # Mean statistic
        mean_func = lambda x: np.mean(x)
        
        # BCa bootstrap
        tests['bca'] = self.bootstrap.bias_corrected_accelerated(excess_returns, mean_func)
        
        # Studentized bootstrap
        variance_func = lambda x: np.var(x, ddof=1) / len(x)
        tests['studentized'] = self.bootstrap.studentized_bootstrap(excess_returns, mean_func, variance_func)
        
        # Block bootstrap (for time series)
        tests['block'] = self.bootstrap.block_bootstrap(excess_returns, mean_func)
        
        return tests
        
    def _bayesian_tests(self, excess_returns: np.ndarray) -> Dict[str, BayesianResult]:
        """Run Bayesian tests"""
        tests = {}
        
        # Bayesian t-test
        tests['t_test'] = self.bayesian.bayesian_t_test(excess_returns, mu_0=0)
        
        # Bayesian proportion test (proportion of positive returns)
        positive_returns = np.sum(excess_returns > 0)
        total_returns = len(excess_returns)
        tests['proportion'] = self.bayesian.bayesian_proportion_test(positive_returns, total_returns, p_0=0.5)
        
        return tests
        
    def _robust_tests(self, excess_returns: np.ndarray) -> Dict[str, TestResult]:
        """Run robust tests"""
        tests = {}
        
        # Trimmed mean test
        tests['trimmed_mean'] = self.robust.trimmed_mean_test(excess_returns, trim_proportion=0.1)
        
        # Median test
        tests['median'] = self.robust.median_test(excess_returns)
        
        return tests
        
    def _permutation_tests(self, portfolio_returns: np.ndarray, benchmark_returns: np.ndarray) -> Dict[str, TestResult]:
        """Run permutation tests"""
        tests = {}
        
        # Prepare data
        min_len = min(len(portfolio_returns), len(benchmark_returns))
        portfolio_returns = portfolio_returns[:min_len]
        benchmark_returns = benchmark_returns[:min_len]
        
        # Permutation test for mean difference
        observed_diff = np.mean(portfolio_returns) - np.mean(benchmark_returns)
        combined_returns = np.concatenate([portfolio_returns, benchmark_returns])
        
        permutation_diffs = []
        for _ in range(self.n_bootstrap):
            permuted = np.random.permutation(combined_returns)
            perm_portfolio = permuted[:len(portfolio_returns)]
            perm_benchmark = permuted[len(portfolio_returns):]
            permutation_diffs.append(np.mean(perm_portfolio) - np.mean(perm_benchmark))
        
        p_value = np.mean(np.abs(permutation_diffs) >= np.abs(observed_diff))
        
        tests['mean_difference'] = TestResult(
            test_name='Permutation Test (Mean Difference)',
            statistic=observed_diff,
            p_value=p_value,
            critical_value=None,
            confidence_interval=None,
            effect_size=None,
            power=None,
            reject_null=p_value < self.significance_level,
            interpretation="Non-parametric test for difference in means",
            sample_size=len(portfolio_returns) + len(benchmark_returns),
            additional_stats={
                'permutation_distribution': permutation_diffs,
                'observed_difference': observed_diff
            }
        )
        
        return tests
        
    def _multiple_testing_corrections(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Apply multiple testing corrections"""
        corrections = {}
        
        # Collect p-values from all tests
        p_values = []
        test_names = []
        
        for test_category, test_results in results.items():
            if test_category in ['classical', 'robust', 'permutation']:
                for test_name, test_result in test_results.items():
                    p_values.append(test_result.p_value)
                    test_names.append(f"{test_category}_{test_name}")
            elif test_category == 'bootstrap':
                for test_name, test_result in test_results.items():
                    p_values.append(test_result.p_value)
                    test_names.append(f"{test_category}_{test_name}")
        
        if not p_values:
            return {'error': 'No p-values found for correction'}
        
        p_values = np.array(p_values)
        
        # Apply different correction methods
        methods = ['bonferroni', 'holm', 'fdr_bh', 'fdr_by']
        
        for method in methods:
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
        
    def _power_analysis(self, excess_returns: np.ndarray) -> Dict[str, Any]:
        """Perform power analysis"""
        power_analysis = {}
        
        # Observed effect size
        effect_size = np.mean(excess_returns) / np.std(excess_returns, ddof=1)
        n = len(excess_returns)
        
        # Power for current sample size
        from scipy.stats import norm
        
        # For one-sample t-test
        delta = effect_size * np.sqrt(n)
        power = 1 - stats.t.cdf(stats.t.ppf(1 - self.significance_level/2, n-1), n-1, delta) + \
                stats.t.cdf(-stats.t.ppf(1 - self.significance_level/2, n-1), n-1, delta)
        
        power_analysis['current_power'] = power
        power_analysis['effect_size'] = effect_size
        power_analysis['sample_size'] = n
        
        # Required sample size for different power levels
        target_powers = [0.8, 0.9, 0.95]
        required_n = {}
        
        for target_power in target_powers:
            # Approximate formula for required sample size
            if effect_size != 0:
                z_alpha = norm.ppf(1 - self.significance_level/2)
                z_beta = norm.ppf(target_power)
                n_required = ((z_alpha + z_beta) / effect_size) ** 2
                required_n[f'power_{target_power}'] = int(np.ceil(n_required))
            else:
                required_n[f'power_{target_power}'] = float('inf')
        
        power_analysis['required_sample_sizes'] = required_n
        
        return power_analysis
        
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary and recommendations"""
        summary = {}
        
        # Count significant tests
        significant_tests = []
        total_tests = 0
        
        for test_category, test_results in results.items():
            if test_category in ['classical', 'robust', 'permutation']:
                for test_name, test_result in test_results.items():
                    total_tests += 1
                    if test_result.reject_null:
                        significant_tests.append(f"{test_category}_{test_name}")
            elif test_category == 'bootstrap':
                for test_name, test_result in test_results.items():
                    total_tests += 1
                    if test_result.p_value < self.significance_level:
                        significant_tests.append(f"{test_category}_{test_name}")
        
        summary['significant_tests'] = significant_tests
        summary['total_tests'] = total_tests
        summary['proportion_significant'] = len(significant_tests) / total_tests if total_tests > 0 else 0
        
        # Check multiple testing corrections
        if 'multiple_testing' in results:
            mt_results = results['multiple_testing']
            summary['bonferroni_significant'] = mt_results.get('bonferroni', {}).get('any_significant', False)
            summary['fdr_significant'] = mt_results.get('fdr_bh', {}).get('any_significant', False)
        
        # Overall recommendation
        if len(significant_tests) >= 3 and summary.get('fdr_significant', False):
            summary['recommendation'] = 'Strong evidence of significant alpha'
            summary['confidence'] = 'high'
        elif len(significant_tests) >= 2 and summary.get('bonferroni_significant', False):
            summary['recommendation'] = 'Moderate evidence of significant alpha'
            summary['confidence'] = 'medium'
        elif len(significant_tests) >= 1:
            summary['recommendation'] = 'Weak evidence of significant alpha'
            summary['confidence'] = 'low'
        else:
            summary['recommendation'] = 'No evidence of significant alpha'
            summary['confidence'] = 'none'
        
        return summary


def main():
    """Example usage of statistical testing suite"""
    logging.basicConfig(level=logging.INFO)
    
    # Generate synthetic data
    np.random.seed(42)
    n = 1000
    
    # Portfolio with small positive alpha
    portfolio_returns = np.random.normal(0.001, 0.02, n)  # 0.1% daily alpha
    benchmark_returns = np.random.normal(0.0005, 0.015, n)  # 0.05% daily return
    
    # Initialize test suite
    test_suite = StatisticalTestSuite(significance_level=0.05, n_bootstrap=1000)
    
    # Run comprehensive tests
    results = test_suite.comprehensive_alpha_test(portfolio_returns, benchmark_returns)
    
    # Print results
    print("=" * 60)
    print("COMPREHENSIVE ALPHA TESTING RESULTS")
    print("=" * 60)
    
    # Classical tests
    print("\nCLASSICAL TESTS:")
    for test_name, test_result in results['classical'].items():
        print(f"{test_name}: p-value = {test_result.p_value:.4f}, reject = {test_result.reject_null}")
    
    # Bootstrap tests
    print("\nBOOTSTRAP TESTS:")
    for test_name, test_result in results['bootstrap'].items():
        print(f"{test_name}: p-value = {test_result.p_value:.4f}, CI = {test_result.confidence_interval}")
    
    # Multiple testing corrections
    print("\nMULTIPLE TESTING CORRECTIONS:")
    if 'multiple_testing' in results:
        mt = results['multiple_testing']
        for method, correction in mt.items():
            if 'error' not in correction:
                print(f"{method}: {correction['num_significant']}/{len(correction['rejected'])} significant")
    
    # Summary
    print("\nSUMMARY:")
    summary = results['summary']
    print(f"Significant tests: {len(summary['significant_tests'])}/{summary['total_tests']}")
    print(f"Recommendation: {summary['recommendation']}")
    print(f"Confidence: {summary['confidence']}")


if __name__ == "__main__":
    main()