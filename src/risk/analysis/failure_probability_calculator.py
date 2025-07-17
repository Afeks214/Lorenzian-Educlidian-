"""
Failure Probability Calculator for Pre-Mortem Analysis

Calculates failure probabilities and generates GO/CAUTION/NO-GO recommendations
based on Monte Carlo simulation results and advanced risk metrics.

Key Features:
- 3-tier recommendation system (GO/CAUTION/NO-GO)
- Multiple risk metrics (VaR, Expected Shortfall, Max Drawdown)
- Configurable failure thresholds
- Statistical validation and confidence intervals
- Human review trigger logic
- Real-time performance monitoring
"""

import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import structlog
from scipy import stats

from src.risk.simulation.monte_carlo_engine import SimulationResults

logger = structlog.get_logger()


class RiskRecommendation(Enum):
    """Risk-based decision recommendations"""
    GO = "GO"                           # <5% failure probability - proceed with confidence
    GO_WITH_CAUTION = "CAUTION"        # 5-15% failure probability - proceed with monitoring
    NO_GO_REQUIRES_HUMAN_REVIEW = "NO_GO"  # >15% failure probability - halt and escalate


@dataclass
class FailureMetrics:
    """Comprehensive failure probability metrics"""
    # Primary metrics
    failure_probability: float = 0.0           # Overall failure probability
    var_95_percent: float = 0.0                # Value at Risk (95%)
    var_99_percent: float = 0.0                # Value at Risk (99%)
    expected_shortfall_95: float = 0.0         # Expected Shortfall (95%)
    expected_shortfall_99: float = 0.0         # Expected Shortfall (99%)
    max_drawdown_probability: float = 0.0      # Probability of significant drawdown
    
    # Distribution metrics
    return_mean: float = 0.0                   # Mean portfolio return
    return_std: float = 0.0                    # Standard deviation of returns
    return_skewness: float = 0.0               # Return distribution skewness
    return_kurtosis: float = 0.0               # Return distribution kurtosis
    
    # Time-based metrics
    time_to_recovery_days: float = 0.0         # Expected recovery time
    worst_case_recovery_days: float = 0.0      # 95th percentile recovery time
    
    # Probability metrics  
    prob_loss_2pct: float = 0.0               # Probability of >2% loss
    prob_loss_5pct: float = 0.0               # Probability of >5% loss
    prob_loss_10pct: float = 0.0              # Probability of >10% loss
    prob_gain_2pct: float = 0.0               # Probability of >2% gain
    prob_gain_5pct: float = 0.0               # Probability of >5% gain
    
    # Confidence intervals
    failure_prob_lower_ci: float = 0.0        # Lower confidence interval
    failure_prob_upper_ci: float = 0.0        # Upper confidence interval
    confidence_level: float = 0.95            # Confidence level
    
    # Recommendation support
    recommendation: RiskRecommendation = RiskRecommendation.GO
    recommendation_confidence: float = 0.0    # Confidence in recommendation
    requires_human_review: bool = False       # Human review flag
    human_review_reasons: List[str] = field(default_factory=list)
    
    # Risk attribution
    primary_risk_factors: List[str] = field(default_factory=list)
    risk_concentration_score: float = 0.0    # Risk concentration metric
    
    # Performance metadata
    calculation_time_ms: float = 0.0         # Calculation time


@dataclass
class FailureThresholds:
    """Configurable thresholds for failure analysis"""
    # Primary failure definition (portfolio loss thresholds)
    primary_loss_threshold: float = 0.02      # 2% portfolio loss
    secondary_loss_threshold: float = 0.05    # 5% portfolio loss
    severe_loss_threshold: float = 0.10       # 10% portfolio loss
    
    # Recommendation thresholds
    go_max_failure_prob: float = 0.05         # <5% for GO
    caution_max_failure_prob: float = 0.15    # 5-15% for CAUTION
    # >15% triggers NO_GO
    
    # Drawdown thresholds
    max_drawdown_threshold: float = 0.15      # 15% max drawdown
    drawdown_prob_threshold: float = 0.20     # 20% probability threshold
    
    # VaR thresholds
    var_95_threshold: float = 0.03            # 3% VaR threshold
    var_99_threshold: float = 0.05            # 5% VaR threshold
    
    # Human review triggers
    human_review_failure_prob: float = 0.12   # 12% failure probability
    human_review_var_threshold: float = 0.04  # 4% VaR
    human_review_expected_shortfall: float = 0.06  # 6% Expected Shortfall
    
    # Statistical significance
    min_confidence_level: float = 0.95        # Minimum confidence level
    min_simulation_paths: int = 1000          # Minimum paths for reliable estimates


class FailureProbabilityCalculator:
    """
    Advanced Failure Probability Calculator
    
    Analyzes Monte Carlo simulation results to calculate failure probabilities
    and generate risk-based trading recommendations with statistical validation.
    
    Features:
    - Multi-threshold failure analysis
    - Statistical confidence intervals
    - Risk attribution analysis
    - Human review triggers
    - Performance optimization
    """
    
    def __init__(self, 
                 thresholds: Optional[FailureThresholds] = None,
                 confidence_level: float = 0.95):
        """
        Initialize failure probability calculator
        
        Args:
            thresholds: Custom failure thresholds (uses defaults if None)
            confidence_level: Statistical confidence level for intervals
        """
        self.thresholds = thresholds or FailureThresholds()
        self.confidence_level = confidence_level
        
        # Performance tracking
        self.calculation_times = []
        self.recommendations_history = []
        
        logger.info("Failure probability calculator initialized",
                   primary_threshold=self.thresholds.primary_loss_threshold,
                   confidence_level=confidence_level)
    
    def calculate_failure_metrics(self, 
                                 simulation_results: SimulationResults,
                                 current_portfolio_value: float = 1.0) -> FailureMetrics:
        """
        Calculate comprehensive failure metrics from simulation results
        
        Args:
            simulation_results: Monte Carlo simulation results
            current_portfolio_value: Current portfolio value (normalized to 1.0)
            
        Returns:
            Comprehensive failure metrics and recommendation
        """
        start_time = time.perf_counter()
        
        try:
            # Validate simulation results
            self._validate_simulation_results(simulation_results)
            
            # Extract portfolio returns
            final_returns = (simulation_results.final_portfolio_values - current_portfolio_value)
            return_pct = final_returns / current_portfolio_value
            
            # Calculate basic failure probabilities
            failure_metrics = self._calculate_basic_metrics(return_pct, simulation_results)
            
            # Calculate VaR and Expected Shortfall
            self._calculate_var_metrics(failure_metrics, return_pct)
            
            # Calculate drawdown metrics
            self._calculate_drawdown_metrics(failure_metrics, simulation_results)
            
            # Calculate distribution characteristics
            self._calculate_distribution_metrics(failure_metrics, return_pct)
            
            # Calculate time-based metrics
            self._calculate_time_metrics(failure_metrics, simulation_results)
            
            # Calculate confidence intervals
            self._calculate_confidence_intervals(failure_metrics, return_pct)
            
            # Generate recommendation
            self._generate_recommendation(failure_metrics)
            
            # Risk attribution analysis
            self._analyze_risk_factors(failure_metrics, simulation_results)
            
            # Track performance
            calculation_time = (time.perf_counter() - start_time) * 1000
            failure_metrics.calculation_time_ms = calculation_time
            self.calculation_times.append(calculation_time)
            self.recommendations_history.append(failure_metrics.recommendation)
            
            logger.info("Failure metrics calculated",
                       failure_probability=f"{failure_metrics.failure_probability:.3f}",
                       recommendation=failure_metrics.recommendation.value,
                       calculation_time_ms=f"{calculation_time:.2f}")
            
            return failure_metrics
            
        except Exception as e:
            logger.error("Error calculating failure metrics", error=str(e))
            raise
    
    def _validate_simulation_results(self, results: SimulationResults) -> None:
        """Validate simulation results for analysis"""
        if results.final_portfolio_values is None or len(results.final_portfolio_values) == 0:
            raise ValueError("Simulation results contain no portfolio values")
        
        if len(results.final_portfolio_values) < self.thresholds.min_simulation_paths:
            logger.warning("Insufficient simulation paths for reliable estimates",
                          actual=len(results.final_portfolio_values),
                          minimum=self.thresholds.min_simulation_paths)
        
        if np.any(np.isnan(results.final_portfolio_values)):
            raise ValueError("Simulation results contain NaN values")
        
        if np.any(results.final_portfolio_values <= 0):
            logger.warning("Simulation contains non-positive portfolio values")
    
    def _calculate_basic_metrics(self, 
                                return_pct: np.ndarray, 
                                results: SimulationResults) -> FailureMetrics:
        """Calculate basic failure probability metrics"""
        metrics = FailureMetrics()
        
        # Primary failure probability (>2% loss)
        loss_2pct_mask = return_pct <= -self.thresholds.primary_loss_threshold
        metrics.failure_probability = np.mean(loss_2pct_mask)
        
        # Various loss thresholds
        metrics.prob_loss_2pct = np.mean(return_pct <= -0.02)
        metrics.prob_loss_5pct = np.mean(return_pct <= -0.05)
        metrics.prob_loss_10pct = np.mean(return_pct <= -0.10)
        
        # Gain probabilities
        metrics.prob_gain_2pct = np.mean(return_pct >= 0.02)
        metrics.prob_gain_5pct = np.mean(return_pct >= 0.05)
        
        return metrics
    
    def _calculate_var_metrics(self, metrics: FailureMetrics, return_pct: np.ndarray) -> None:
        """Calculate Value at Risk and Expected Shortfall metrics"""
        # VaR calculations (negative values represent losses)
        metrics.var_95_percent = -np.percentile(return_pct, 5)  # 95% VaR
        metrics.var_99_percent = -np.percentile(return_pct, 1)  # 99% VaR
        
        # Expected Shortfall (Conditional VaR)
        var_95_threshold = np.percentile(return_pct, 5)
        var_99_threshold = np.percentile(return_pct, 1)
        
        tail_losses_95 = return_pct[return_pct <= var_95_threshold]
        tail_losses_99 = return_pct[return_pct <= var_99_threshold]
        
        metrics.expected_shortfall_95 = -np.mean(tail_losses_95) if len(tail_losses_95) > 0 else 0.0
        metrics.expected_shortfall_99 = -np.mean(tail_losses_99) if len(tail_losses_99) > 0 else 0.0
    
    def _calculate_drawdown_metrics(self, 
                                   metrics: FailureMetrics, 
                                   results: SimulationResults) -> None:
        """Calculate maximum drawdown probability metrics"""
        if results.max_drawdowns is None:
            return
        
        # Probability of significant drawdown
        significant_dd_mask = results.max_drawdowns >= self.thresholds.max_drawdown_threshold
        metrics.max_drawdown_probability = np.mean(significant_dd_mask)
    
    def _calculate_distribution_metrics(self, 
                                       metrics: FailureMetrics, 
                                       return_pct: np.ndarray) -> None:
        """Calculate return distribution characteristics"""
        metrics.return_mean = np.mean(return_pct)
        metrics.return_std = np.std(return_pct)
        metrics.return_skewness = stats.skew(return_pct)
        metrics.return_kurtosis = stats.kurtosis(return_pct)
    
    def _calculate_time_metrics(self, 
                               metrics: FailureMetrics, 
                               results: SimulationResults) -> None:
        """Calculate time-based recovery metrics"""
        # Simplified recovery time estimation
        # In practice, would analyze portfolio value paths over time
        
        if metrics.return_mean > 0:
            # Estimate recovery time based on mean return
            daily_return = metrics.return_mean / (24.0 / 24.0)  # Convert hourly to daily
            loss_amount = metrics.var_95_percent
            
            if daily_return > 0:
                metrics.time_to_recovery_days = loss_amount / daily_return
                metrics.worst_case_recovery_days = metrics.time_to_recovery_days * 2  # Conservative estimate
            else:
                metrics.time_to_recovery_days = float('inf')
                metrics.worst_case_recovery_days = float('inf')
    
    def _calculate_confidence_intervals(self, 
                                       metrics: FailureMetrics, 
                                       return_pct: np.ndarray) -> None:
        """Calculate confidence intervals for failure probability"""
        n_simulations = len(return_pct)
        
        # Binomial confidence interval for failure probability
        p = metrics.failure_probability
        
        # Wilson score interval (more robust than normal approximation)
        z = stats.norm.ppf(1 - (1 - self.confidence_level) / 2)
        
        denominator = 1 + z**2 / n_simulations
        center = p + z**2 / (2 * n_simulations)
        spread = z * np.sqrt(p * (1 - p) / n_simulations + z**2 / (4 * n_simulations**2))
        
        metrics.failure_prob_lower_ci = max(0, (center - spread) / denominator)
        metrics.failure_prob_upper_ci = min(1, (center + spread) / denominator)
        metrics.confidence_level = self.confidence_level
    
    def _generate_recommendation(self, metrics: FailureMetrics) -> None:
        """Generate risk recommendation based on failure metrics"""
        failure_prob = metrics.failure_probability
        
        # Primary recommendation logic
        if failure_prob <= self.thresholds.go_max_failure_prob:
            metrics.recommendation = RiskRecommendation.GO
            metrics.recommendation_confidence = 1.0 - failure_prob
        elif failure_prob <= self.thresholds.caution_max_failure_prob:
            metrics.recommendation = RiskRecommendation.GO_WITH_CAUTION
            metrics.recommendation_confidence = 1.0 - (failure_prob - self.thresholds.go_max_failure_prob) / \
                (self.thresholds.caution_max_failure_prob - self.thresholds.go_max_failure_prob)
        else:
            metrics.recommendation = RiskRecommendation.NO_GO_REQUIRES_HUMAN_REVIEW
            metrics.recommendation_confidence = min(1.0, failure_prob - self.thresholds.caution_max_failure_prob)
        
        # Check for human review triggers
        self._check_human_review_triggers(metrics)
        
        # Apply additional safety checks
        self._apply_safety_overrides(metrics)
    
    def _check_human_review_triggers(self, metrics: FailureMetrics) -> None:
        """Check conditions that trigger human review"""
        review_reasons = []
        
        # High failure probability
        if metrics.failure_probability >= self.thresholds.human_review_failure_prob:
            review_reasons.append(f"High failure probability ({metrics.failure_probability:.2%})")
        
        # High VaR
        if metrics.var_95_percent >= self.thresholds.human_review_var_threshold:
            review_reasons.append(f"High VaR 95% ({metrics.var_95_percent:.2%})")
        
        # High Expected Shortfall
        if metrics.expected_shortfall_95 >= self.thresholds.human_review_expected_shortfall:
            review_reasons.append(f"High Expected Shortfall ({metrics.expected_shortfall_95:.2%})")
        
        # Extreme distribution characteristics
        if abs(metrics.return_skewness) > 2.0:
            review_reasons.append(f"Extreme return skewness ({metrics.return_skewness:.2f})")
        
        if metrics.return_kurtosis > 10.0:
            review_reasons.append(f"High return kurtosis ({metrics.return_kurtosis:.2f})")
        
        # Wide confidence intervals (uncertain estimates)
        ci_width = metrics.failure_prob_upper_ci - metrics.failure_prob_lower_ci
        if ci_width > 0.1:  # 10% width threshold
            review_reasons.append(f"Wide confidence interval ({ci_width:.2%})")
        
        if review_reasons:
            metrics.requires_human_review = True
            metrics.human_review_reasons = review_reasons
    
    def _apply_safety_overrides(self, metrics: FailureMetrics) -> None:
        """Apply safety overrides to recommendations"""
        # Conservative override for extreme scenarios
        if metrics.var_99_percent > 0.20:  # 20% VaR 99%
            metrics.recommendation = RiskRecommendation.NO_GO_REQUIRES_HUMAN_REVIEW
            metrics.requires_human_review = True
            metrics.human_review_reasons.append("Extreme VaR 99% scenario")
        
        # Override for high drawdown probability
        if metrics.max_drawdown_probability > 0.50:  # 50% chance of significant drawdown
            if metrics.recommendation == RiskRecommendation.GO:
                metrics.recommendation = RiskRecommendation.GO_WITH_CAUTION
        
        # Conservative adjustment for low confidence
        if metrics.recommendation_confidence < 0.5:
            if metrics.recommendation == RiskRecommendation.GO:
                metrics.recommendation = RiskRecommendation.GO_WITH_CAUTION
    
    def _analyze_risk_factors(self, 
                             metrics: FailureMetrics, 
                             results: SimulationResults) -> None:
        """Analyze primary risk factors contributing to failure probability"""
        risk_factors = []
        
        # High volatility risk
        if metrics.return_std > 0.15:  # 15% volatility threshold
            risk_factors.append("High portfolio volatility")
        
        # Negative skew risk (tail risk)
        if metrics.return_skewness < -1.0:
            risk_factors.append("Negative return skewness (tail risk)")
        
        # Fat tail risk
        if metrics.return_kurtosis > 5.0:
            risk_factors.append("Fat tail distribution")
        
        # VaR concentration risk
        if metrics.var_95_percent > metrics.return_std * 2:
            risk_factors.append("VaR concentration risk")
        
        # Drawdown risk
        if metrics.max_drawdown_probability > 0.3:
            risk_factors.append("High drawdown probability")
        
        metrics.primary_risk_factors = risk_factors
        
        # Calculate risk concentration score
        metrics.risk_concentration_score = min(1.0, len(risk_factors) / 5.0)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get calculator performance statistics"""
        if not self.calculation_times:
            return {}
        
        recommendations_count = {rec.value: 0 for rec in RiskRecommendation}
        for rec in self.recommendations_history:
            recommendations_count[rec.value] += 1
        
        return {
            'avg_calculation_time_ms': np.mean(self.calculation_times),
            'max_calculation_time_ms': np.max(self.calculation_times),
            'min_calculation_time_ms': np.min(self.calculation_times),
            'total_calculations': len(self.calculation_times),
            'recommendations_distribution': recommendations_count,
            'go_recommendation_rate': recommendations_count['GO'] / max(1, len(self.recommendations_history)),
            'human_review_rate': recommendations_count['NO_GO'] / max(1, len(self.recommendations_history))
        }
    
    def benchmark_performance(self, n_tests: int = 100) -> Dict[str, Any]:
        """Run performance benchmark"""
        logger.info(f"Running failure probability calculator benchmark ({n_tests} tests)")
        
        # Generate synthetic simulation results for benchmarking
        benchmark_times = []
        
        for _ in range(n_tests):
            # Create synthetic simulation results
            num_paths = 10000
            final_values = np.random.normal(1.05, 0.15, num_paths)  # 5% return, 15% vol
            final_values = np.maximum(final_values, 0.01)  # Ensure positive
            
            max_drawdowns = np.random.beta(2, 8, num_paths) * 0.5  # Beta distribution for drawdowns
            
            synthetic_results = SimulationResults(
                price_paths=None,
                return_paths=None,
                portfolio_values=None,
                final_portfolio_values=final_values,
                max_drawdowns=max_drawdowns,
                computation_time_ms=0.0
            )
            
            start_time = time.perf_counter()
            self.calculate_failure_metrics(synthetic_results)
            benchmark_times.append((time.perf_counter() - start_time) * 1000)
        
        benchmark_stats = {
            'avg_time_ms': np.mean(benchmark_times),
            'min_time_ms': np.min(benchmark_times),
            'max_time_ms': np.max(benchmark_times),
            'std_time_ms': np.std(benchmark_times),
            'percentile_95_ms': np.percentile(benchmark_times, 95),
            'calculations_per_second': 1000 / np.mean(benchmark_times)
        }
        
        logger.info("Benchmark completed", **benchmark_stats)
        return benchmark_stats