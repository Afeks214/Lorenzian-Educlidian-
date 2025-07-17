"""
Comprehensive Risk Validation and Testing Framework
===================================================

This module implements a comprehensive risk validation and testing framework
that includes:

- Risk model validation and backtesting
- Statistical testing for risk measures
- Model performance validation
- Stress testing and scenario analysis
- Regulatory compliance validation
- Integration testing for all risk components

Author: Risk Management System
Date: 2025-07-17
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio
import logging
from scipy import stats
from scipy.stats import kstest, jarque_bera, normaltest
from numba import jit, njit
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    """Configuration for risk validation"""
    # Backtesting parameters
    backtesting_window: int = 252
    confidence_levels: List[float] = field(default_factory=lambda: [0.95, 0.99])
    var_methods: List[str] = field(default_factory=lambda: ["historical", "parametric", "monte_carlo"])
    
    # Statistical tests
    normality_test_enabled: bool = True
    independence_test_enabled: bool = True
    stationarity_test_enabled: bool = True
    
    # Performance validation
    performance_threshold: float = 0.05
    accuracy_threshold: float = 0.95
    coverage_threshold: float = 0.90
    
    # Stress testing
    stress_scenarios: List[str] = field(default_factory=lambda: [
        "market_crash", "volatility_spike", "correlation_breakdown"
    ])
    
    # Regulatory compliance
    basel_validation: bool = True
    var_exceptions_threshold: int = 5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'backtesting_window': self.backtesting_window,
            'confidence_levels': self.confidence_levels,
            'var_methods': self.var_methods,
            'normality_test_enabled': self.normality_test_enabled,
            'independence_test_enabled': self.independence_test_enabled,
            'stationarity_test_enabled': self.stationarity_test_enabled,
            'performance_threshold': self.performance_threshold,
            'accuracy_threshold': self.accuracy_threshold,
            'coverage_threshold': self.coverage_threshold,
            'stress_scenarios': self.stress_scenarios,
            'basel_validation': self.basel_validation,
            'var_exceptions_threshold': self.var_exceptions_threshold
        }


@dataclass
class BacktestResult:
    """Backtesting result"""
    method: str
    confidence_level: float
    total_observations: int
    exceptions: int
    exception_rate: float
    coverage: float
    independence_test_pvalue: float
    kupiec_test_pvalue: float
    christoffersen_test_pvalue: float
    var_accuracy: float
    performance_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'method': self.method,
            'confidence_level': self.confidence_level,
            'total_observations': self.total_observations,
            'exceptions': self.exceptions,
            'exception_rate': self.exception_rate,
            'coverage': self.coverage,
            'independence_test_pvalue': self.independence_test_pvalue,
            'kupiec_test_pvalue': self.kupiec_test_pvalue,
            'christoffersen_test_pvalue': self.christoffersen_test_pvalue,
            'var_accuracy': self.var_accuracy,
            'performance_score': self.performance_score
        }


@dataclass
class StatisticalTestResult:
    """Statistical test result"""
    test_name: str
    statistic: float
    p_value: float
    critical_value: float
    passed: bool
    interpretation: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'test_name': self.test_name,
            'statistic': self.statistic,
            'p_value': self.p_value,
            'critical_value': self.critical_value,
            'passed': self.passed,
            'interpretation': self.interpretation
        }


@dataclass
class ValidationReport:
    """Comprehensive validation report"""
    validation_date: datetime
    backtest_results: List[BacktestResult]
    statistical_tests: List[StatisticalTestResult]
    stress_test_results: Dict[str, Any]
    performance_metrics: Dict[str, float]
    compliance_status: Dict[str, bool]
    recommendations: List[str]
    overall_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'validation_date': self.validation_date.isoformat(),
            'backtest_results': [result.to_dict() for result in self.backtest_results],
            'statistical_tests': [test.to_dict() for test in self.statistical_tests],
            'stress_test_results': self.stress_test_results,
            'performance_metrics': self.performance_metrics,
            'compliance_status': self.compliance_status,
            'recommendations': self.recommendations,
            'overall_score': self.overall_score
        }


# JIT optimized validation functions
@njit(cache=True, fastmath=True)
def calculate_var_exceptions(returns: np.ndarray, var_forecasts: np.ndarray) -> np.ndarray:
    """
    Calculate VaR exceptions - JIT optimized
    
    Args:
        returns: Actual returns
        var_forecasts: VaR forecasts
    
    Returns:
        Binary array of exceptions (1 = exception, 0 = no exception)
    """
    exceptions = np.zeros(len(returns))
    
    for i in range(len(returns)):
        if returns[i] < -var_forecasts[i]:
            exceptions[i] = 1.0
    
    return exceptions


@njit(cache=True, fastmath=True)
def calculate_kupiec_test_statistic(
    exceptions: np.ndarray,
    confidence_level: float
) -> float:
    """
    Calculate Kupiec test statistic - JIT optimized
    
    Args:
        exceptions: Binary array of exceptions
        confidence_level: VaR confidence level
    
    Returns:
        Test statistic
    """
    n = len(exceptions)
    x = np.sum(exceptions)
    p = 1 - confidence_level
    
    if x == 0:
        return 0.0
    
    if x == n:
        return np.inf
    
    # Kupiec likelihood ratio test statistic
    likelihood_ratio = (x / n) ** x * ((n - x) / n) ** (n - x) / (p ** x * (1 - p) ** (n - x))
    
    if likelihood_ratio <= 0:
        return 0.0
    
    test_statistic = -2 * np.log(likelihood_ratio)
    
    return test_statistic


@njit(cache=True, fastmath=True)
def calculate_christoffersen_test_statistic(exceptions: np.ndarray) -> float:
    """
    Calculate Christoffersen independence test statistic - JIT optimized
    
    Args:
        exceptions: Binary array of exceptions
    
    Returns:
        Test statistic
    """
    n = len(exceptions)
    
    if n < 2:
        return 0.0
    
    # Count transitions
    n00 = 0  # No exception followed by no exception
    n01 = 0  # No exception followed by exception
    n10 = 0  # Exception followed by no exception
    n11 = 0  # Exception followed by exception
    
    for i in range(n - 1):
        if exceptions[i] == 0 and exceptions[i + 1] == 0:
            n00 += 1
        elif exceptions[i] == 0 and exceptions[i + 1] == 1:
            n01 += 1
        elif exceptions[i] == 1 and exceptions[i + 1] == 0:
            n10 += 1
        elif exceptions[i] == 1 and exceptions[i + 1] == 1:
            n11 += 1
    
    # Calculate transition probabilities
    n0 = n00 + n01
    n1 = n10 + n11
    
    if n0 == 0 or n1 == 0:
        return 0.0
    
    p01 = n01 / n0 if n0 > 0 else 0
    p11 = n11 / n1 if n1 > 0 else 0
    p = (n01 + n11) / (n - 1) if n > 1 else 0
    
    if p01 == 0 or p11 == 0 or p == 0:
        return 0.0
    
    # Likelihood ratio
    if p01 == p11:
        return 0.0
    
    likelihood_ratio = (p01 ** n01 * (1 - p01) ** n00 * p11 ** n11 * (1 - p11) ** n10) / (p ** (n01 + n11) * (1 - p) ** (n00 + n10))
    
    if likelihood_ratio <= 0:
        return 0.0
    
    test_statistic = -2 * np.log(likelihood_ratio)
    
    return test_statistic


@njit(cache=True, fastmath=True)
def calculate_coverage_probability(exceptions: np.ndarray, confidence_level: float) -> float:
    """
    Calculate coverage probability - JIT optimized
    
    Args:
        exceptions: Binary array of exceptions
        confidence_level: VaR confidence level
    
    Returns:
        Coverage probability
    """
    n = len(exceptions)
    x = np.sum(exceptions)
    
    expected_exceptions = (1 - confidence_level) * n
    actual_coverage = 1 - (x / n)
    
    return actual_coverage


class RiskValidationFramework:
    """
    Comprehensive Risk Validation Framework
    
    This class implements sophisticated risk model validation including:
    - VaR model backtesting
    - Statistical testing
    - Performance validation
    - Stress testing
    - Regulatory compliance
    """
    
    def __init__(self, config: ValidationConfig):
        """
        Initialize risk validation framework
        
        Args:
            config: Validation configuration
        """
        self.config = config
        
        # Results storage
        self.validation_reports: List[ValidationReport] = []
        self.backtest_results: Dict[str, List[BacktestResult]] = {}
        self.statistical_tests: Dict[str, List[StatisticalTestResult]] = {}
        
        # Performance tracking
        self.validation_times: List[float] = []
        
        logger.info("RiskValidationFramework initialized",
                   extra={'config': config.to_dict()})
    
    async def validate_var_models(
        self,
        returns: np.ndarray,
        var_forecasts: Dict[str, np.ndarray],
        confidence_levels: Optional[List[float]] = None
    ) -> List[BacktestResult]:
        """
        Validate VaR models using backtesting
        
        Args:
            returns: Actual returns
            var_forecasts: VaR forecasts by method
            confidence_levels: Confidence levels to test
        
        Returns:
            List of backtesting results
        """
        start_time = datetime.now()
        
        try:
            if confidence_levels is None:
                confidence_levels = self.config.confidence_levels
            
            backtest_results = []
            
            for method, forecasts in var_forecasts.items():
                for confidence_level in confidence_levels:
                    # Ensure same length
                    min_len = min(len(returns), len(forecasts))
                    returns_subset = returns[:min_len]
                    forecasts_subset = forecasts[:min_len]
                    
                    # Calculate exceptions
                    exceptions = calculate_var_exceptions(returns_subset, forecasts_subset)
                    
                    # Calculate exception rate
                    exception_rate = np.mean(exceptions)
                    
                    # Calculate coverage
                    coverage = calculate_coverage_probability(exceptions, confidence_level)
                    
                    # Kupiec test
                    kupiec_stat = calculate_kupiec_test_statistic(exceptions, confidence_level)
                    kupiec_pvalue = 1 - stats.chi2.cdf(kupiec_stat, df=1)
                    
                    # Christoffersen test
                    christoffersen_stat = calculate_christoffersen_test_statistic(exceptions)
                    christoffersen_pvalue = 1 - stats.chi2.cdf(christoffersen_stat, df=1)
                    
                    # Independence test (simplified)
                    independence_pvalue = self._test_independence(exceptions)
                    
                    # VaR accuracy
                    var_accuracy = self._calculate_var_accuracy(returns_subset, forecasts_subset)
                    
                    # Performance score
                    performance_score = self._calculate_performance_score(
                        exception_rate, coverage, kupiec_pvalue, christoffersen_pvalue
                    )
                    
                    # Create result
                    result = BacktestResult(
                        method=method,
                        confidence_level=confidence_level,
                        total_observations=len(returns_subset),
                        exceptions=int(np.sum(exceptions)),
                        exception_rate=exception_rate,
                        coverage=coverage,
                        independence_test_pvalue=independence_pvalue,
                        kupiec_test_pvalue=kupiec_pvalue,
                        christoffersen_test_pvalue=christoffersen_pvalue,
                        var_accuracy=var_accuracy,
                        performance_score=performance_score
                    )
                    
                    backtest_results.append(result)
            
            # Store results
            method_key = f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.backtest_results[method_key] = backtest_results
            
            return backtest_results
            
        except Exception as e:
            logger.error(f"VaR model validation failed: {e}")
            raise
        
        finally:
            # Track performance
            validation_time = (datetime.now() - start_time).total_seconds() * 1000
            self.validation_times.append(validation_time)
    
    def _test_independence(self, exceptions: np.ndarray) -> float:
        """Test independence of exceptions"""
        
        if len(exceptions) < 2:
            return 1.0
        
        # Simplified independence test using runs test
        n = len(exceptions)
        runs = 1
        
        for i in range(1, n):
            if exceptions[i] != exceptions[i-1]:
                runs += 1
        
        # Calculate expected runs and variance
        n1 = np.sum(exceptions)
        n2 = n - n1
        
        if n1 == 0 or n2 == 0:
            return 1.0
        
        expected_runs = (2 * n1 * n2) / n + 1
        variance_runs = (2 * n1 * n2 * (2 * n1 * n2 - n)) / (n * n * (n - 1))
        
        if variance_runs <= 0:
            return 1.0
        
        # Z-statistic
        z_stat = (runs - expected_runs) / np.sqrt(variance_runs)
        
        # Two-tailed p-value
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        return p_value
    
    def _calculate_var_accuracy(self, returns: np.ndarray, forecasts: np.ndarray) -> float:
        """Calculate VaR accuracy"""
        
        # Calculate mean absolute error
        mae = np.mean(np.abs(returns - (-forecasts)))
        
        # Normalize by return volatility
        return_volatility = np.std(returns)
        
        if return_volatility == 0:
            return 0.0
        
        # Accuracy score (lower MAE = higher accuracy)
        accuracy = 1 - (mae / return_volatility)
        
        return max(0.0, accuracy)
    
    def _calculate_performance_score(
        self,
        exception_rate: float,
        coverage: float,
        kupiec_pvalue: float,
        christoffersen_pvalue: float
    ) -> float:
        """Calculate overall performance score"""
        
        # Score components
        exception_score = 1 - abs(exception_rate - 0.05)  # Target 5% exceptions
        coverage_score = coverage
        kupiec_score = 1 if kupiec_pvalue > 0.05 else 0
        christoffersen_score = 1 if christoffersen_pvalue > 0.05 else 0
        
        # Weighted average
        weights = [0.3, 0.3, 0.2, 0.2]
        scores = [exception_score, coverage_score, kupiec_score, christoffersen_score]
        
        performance_score = np.average(scores, weights=weights)
        
        return max(0.0, min(1.0, performance_score))
    
    async def run_statistical_tests(
        self,
        returns: np.ndarray
    ) -> List[StatisticalTestResult]:
        """
        Run statistical tests on returns
        
        Args:
            returns: Return series
        
        Returns:
            List of statistical test results
        """
        
        test_results = []
        
        # Normality tests
        if self.config.normality_test_enabled:
            # Jarque-Bera test
            jb_stat, jb_pvalue = jarque_bera(returns)
            test_results.append(StatisticalTestResult(
                test_name="Jarque-Bera Normality Test",
                statistic=jb_stat,
                p_value=jb_pvalue,
                critical_value=5.99,  # Chi-square critical value at 5%
                passed=jb_pvalue > 0.05,
                interpretation="Normal distribution" if jb_pvalue > 0.05 else "Non-normal distribution"
            ))
            
            # Shapiro-Wilk test (for smaller samples)
            if len(returns) <= 5000:
                sw_stat, sw_pvalue = stats.shapiro(returns)
                test_results.append(StatisticalTestResult(
                    test_name="Shapiro-Wilk Normality Test",
                    statistic=sw_stat,
                    p_value=sw_pvalue,
                    critical_value=0.05,
                    passed=sw_pvalue > 0.05,
                    interpretation="Normal distribution" if sw_pvalue > 0.05 else "Non-normal distribution"
                ))
        
        # Stationarity tests
        if self.config.stationarity_test_enabled:
            # Augmented Dickey-Fuller test
            adf_stat, adf_pvalue, _, _, critical_values, _ = stats.adfuller(returns)
            test_results.append(StatisticalTestResult(
                test_name="Augmented Dickey-Fuller Test",
                statistic=adf_stat,
                p_value=adf_pvalue,
                critical_value=critical_values['5%'],
                passed=adf_pvalue < 0.05,
                interpretation="Stationary" if adf_pvalue < 0.05 else "Non-stationary"
            ))
        
        # Independence tests
        if self.config.independence_test_enabled:
            # Ljung-Box test
            lb_stat, lb_pvalue = stats.acorr_ljungbox(returns, lags=10, return_df=False)
            test_results.append(StatisticalTestResult(
                test_name="Ljung-Box Independence Test",
                statistic=lb_stat.iloc[0] if hasattr(lb_stat, 'iloc') else lb_stat,
                p_value=lb_pvalue.iloc[0] if hasattr(lb_pvalue, 'iloc') else lb_pvalue,
                critical_value=0.05,
                passed=(lb_pvalue.iloc[0] if hasattr(lb_pvalue, 'iloc') else lb_pvalue) > 0.05,
                interpretation="Independent" if (lb_pvalue.iloc[0] if hasattr(lb_pvalue, 'iloc') else lb_pvalue) > 0.05 else "Dependent"
            ))
        
        # Store results
        test_key = f"statistical_tests_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.statistical_tests[test_key] = test_results
        
        return test_results
    
    async def run_stress_tests(
        self,
        returns: np.ndarray,
        weights: np.ndarray
    ) -> Dict[str, Any]:
        """
        Run stress tests on portfolio
        
        Args:
            returns: Return matrix
            weights: Portfolio weights
        
        Returns:
            Stress test results
        """
        
        stress_results = {}
        
        for scenario in self.config.stress_scenarios:
            if scenario == "market_crash":
                # Simulate market crash
                stressed_returns = returns - 0.20  # 20% market drop
                portfolio_loss = np.dot(stressed_returns, weights)
                stress_results[scenario] = {
                    'portfolio_loss': np.sum(portfolio_loss),
                    'max_single_day_loss': np.min(portfolio_loss),
                    'recovery_probability': 0.75
                }
            
            elif scenario == "volatility_spike":
                # Simulate volatility spike
                vol_multiplier = 3.0
                stressed_returns = returns * vol_multiplier
                portfolio_loss = np.dot(stressed_returns, weights)
                stress_results[scenario] = {
                    'portfolio_loss': np.sum(portfolio_loss),
                    'max_single_day_loss': np.min(portfolio_loss),
                    'recovery_probability': 0.85
                }
            
            elif scenario == "correlation_breakdown":
                # Simulate correlation breakdown
                correlation_increase = 0.3
                # Simplified correlation stress
                stressed_returns = returns * (1 + correlation_increase)
                portfolio_loss = np.dot(stressed_returns, weights)
                stress_results[scenario] = {
                    'portfolio_loss': np.sum(portfolio_loss),
                    'max_single_day_loss': np.min(portfolio_loss),
                    'recovery_probability': 0.65
                }
        
        return stress_results
    
    async def validate_compliance(
        self,
        backtest_results: List[BacktestResult]
    ) -> Dict[str, bool]:
        """
        Validate regulatory compliance
        
        Args:
            backtest_results: Backtesting results
        
        Returns:
            Compliance status
        """
        
        compliance_status = {}
        
        if self.config.basel_validation:
            # Basel III compliance
            for result in backtest_results:
                if result.confidence_level == 0.99:
                    # Check exception rate
                    annual_exceptions = result.exceptions * (252 / result.total_observations)
                    compliance_status[f"basel_exceptions_{result.method}"] = annual_exceptions <= self.config.var_exceptions_threshold
                    
                    # Check coverage
                    compliance_status[f"basel_coverage_{result.method}"] = result.coverage >= self.config.coverage_threshold
                    
                    # Check independence
                    compliance_status[f"basel_independence_{result.method}"] = result.independence_test_pvalue > 0.05
        
        return compliance_status
    
    async def generate_validation_report(
        self,
        returns: np.ndarray,
        var_forecasts: Dict[str, np.ndarray],
        weights: Optional[np.ndarray] = None
    ) -> ValidationReport:
        """
        Generate comprehensive validation report
        
        Args:
            returns: Return series
            var_forecasts: VaR forecasts
            weights: Portfolio weights
        
        Returns:
            Validation report
        """
        
        start_time = datetime.now()
        
        try:
            # Run backtesting
            backtest_results = await self.validate_var_models(returns, var_forecasts)
            
            # Run statistical tests
            statistical_tests = await self.run_statistical_tests(returns)
            
            # Run stress tests
            stress_test_results = {}
            if weights is not None:
                stress_test_results = await self.run_stress_tests(returns.reshape(-1, 1), weights)
            
            # Validate compliance
            compliance_status = await self.validate_compliance(backtest_results)
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(backtest_results, statistical_tests)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                backtest_results, statistical_tests, compliance_status
            )
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(
                backtest_results, statistical_tests, compliance_status
            )
            
            # Create validation report
            report = ValidationReport(
                validation_date=datetime.now(),
                backtest_results=backtest_results,
                statistical_tests=statistical_tests,
                stress_test_results=stress_test_results,
                performance_metrics=performance_metrics,
                compliance_status=compliance_status,
                recommendations=recommendations,
                overall_score=overall_score
            )
            
            # Store report
            self.validation_reports.append(report)
            
            return report
            
        except Exception as e:
            logger.error(f"Validation report generation failed: {e}")
            raise
        
        finally:
            # Track performance
            validation_time = (datetime.now() - start_time).total_seconds() * 1000
            self.validation_times.append(validation_time)
    
    def _calculate_performance_metrics(
        self,
        backtest_results: List[BacktestResult],
        statistical_tests: List[StatisticalTestResult]
    ) -> Dict[str, float]:
        """Calculate performance metrics"""
        
        metrics = {}
        
        # Backtesting metrics
        if backtest_results:
            metrics['average_coverage'] = np.mean([r.coverage for r in backtest_results])
            metrics['average_exception_rate'] = np.mean([r.exception_rate for r in backtest_results])
            metrics['average_performance_score'] = np.mean([r.performance_score for r in backtest_results])
            metrics['kupiec_pass_rate'] = np.mean([r.kupiec_test_pvalue > 0.05 for r in backtest_results])
            metrics['christoffersen_pass_rate'] = np.mean([r.christoffersen_test_pvalue > 0.05 for r in backtest_results])
        
        # Statistical test metrics
        if statistical_tests:
            metrics['statistical_test_pass_rate'] = np.mean([t.passed for t in statistical_tests])
        
        return metrics
    
    def _generate_recommendations(
        self,
        backtest_results: List[BacktestResult],
        statistical_tests: List[StatisticalTestResult],
        compliance_status: Dict[str, bool]
    ) -> List[str]:
        """Generate recommendations"""
        
        recommendations = []
        
        # Backtesting recommendations
        for result in backtest_results:
            if result.exception_rate > 0.10:
                recommendations.append(f"High exception rate for {result.method} - consider model recalibration")
            
            if result.coverage < 0.90:
                recommendations.append(f"Low coverage for {result.method} - review model assumptions")
            
            if result.kupiec_test_pvalue < 0.05:
                recommendations.append(f"Kupiec test failed for {result.method} - check model calibration")
            
            if result.christoffersen_test_pvalue < 0.05:
                recommendations.append(f"Independence test failed for {result.method} - check for clustering")
        
        # Statistical test recommendations
        for test in statistical_tests:
            if not test.passed:
                if "normality" in test.test_name.lower():
                    recommendations.append("Non-normal returns detected - consider alternative VaR methods")
                elif "stationarity" in test.test_name.lower():
                    recommendations.append("Non-stationary returns detected - consider time-varying models")
                elif "independence" in test.test_name.lower():
                    recommendations.append("Dependent returns detected - consider GARCH models")
        
        # Compliance recommendations
        for compliance_check, passed in compliance_status.items():
            if not passed:
                recommendations.append(f"Compliance issue: {compliance_check} - review model parameters")
        
        return recommendations
    
    def _calculate_overall_score(
        self,
        backtest_results: List[BacktestResult],
        statistical_tests: List[StatisticalTestResult],
        compliance_status: Dict[str, bool]
    ) -> float:
        """Calculate overall validation score"""
        
        scores = []
        
        # Backtesting score
        if backtest_results:
            backtest_score = np.mean([r.performance_score for r in backtest_results])
            scores.append(backtest_score * 0.5)  # 50% weight
        
        # Statistical test score
        if statistical_tests:
            statistical_score = np.mean([1.0 if t.passed else 0.0 for t in statistical_tests])
            scores.append(statistical_score * 0.3)  # 30% weight
        
        # Compliance score
        if compliance_status:
            compliance_score = np.mean([1.0 if passed else 0.0 for passed in compliance_status.values()])
            scores.append(compliance_score * 0.2)  # 20% weight
        
        # Overall score
        overall_score = np.sum(scores) if scores else 0.0
        
        return max(0.0, min(1.0, overall_score))
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get comprehensive validation summary"""
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_validations': len(self.validation_reports),
            'latest_report': self.validation_reports[-1].to_dict() if self.validation_reports else {},
            'performance_metrics': {
                'avg_validation_time_ms': np.mean(self.validation_times) if self.validation_times else 0,
                'validation_count': len(self.validation_times)
            },
            'historical_scores': [report.overall_score for report in self.validation_reports],
            'config': self.config.to_dict()
        }
        
        return summary


# Factory function
def create_validation_framework(config_dict: Optional[Dict[str, Any]] = None) -> RiskValidationFramework:
    """
    Create a risk validation framework with configuration
    
    Args:
        config_dict: Configuration dictionary
    
    Returns:
        RiskValidationFramework instance
    """
    
    if config_dict is None:
        config = ValidationConfig()
    else:
        config = ValidationConfig(**config_dict)
    
    return RiskValidationFramework(config)