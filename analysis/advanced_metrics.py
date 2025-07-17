"""
Advanced Performance Metrics with VaR/CVaR Integration and Bootstrap Confidence Intervals

This module provides advanced risk-adjusted performance metrics with:
- VaR/CVaR integration using existing risk system
- Bootstrap confidence intervals for all metrics
- Streaming metrics for real-time calculations
- Numba JIT compilation for performance optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
from numba import jit, njit
from scipy import stats
import asyncio
import logging
from datetime import datetime, timedelta

# Suppress warnings from numba
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceInterval:
    """Confidence interval for a metric"""
    lower: float
    upper: float
    confidence_level: float
    
    def __str__(self) -> str:
        return f"[{self.lower:.4f}, {self.upper:.4f}] ({self.confidence_level:.0%})"


@dataclass
class MetricWithConfidence:
    """Metric value with confidence interval"""
    value: float
    confidence_interval: ConfidenceInterval
    bootstrap_samples: int
    
    def __str__(self) -> str:
        return f"{self.value:.4f} {self.confidence_interval}"


@dataclass
class StreamingMetric:
    """Streaming metric that updates incrementally"""
    value: float
    count: int
    last_update: datetime
    
    def update(self, new_value: float) -> None:
        """Update the streaming metric"""
        self.count += 1
        self.value = (self.value * (self.count - 1) + new_value) / self.count
        self.last_update = datetime.now()


@dataclass
class RiskAdjustedMetrics:
    """Container for risk-adjusted metrics with VaR/CVaR"""
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    var_adjusted_return: float
    cvar_adjusted_return: float
    risk_adjusted_alpha: float
    downside_capture_ratio: float
    upside_capture_ratio: float
    capture_ratio: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'var_adjusted_return': self.var_adjusted_return,
            'cvar_adjusted_return': self.cvar_adjusted_return,
            'risk_adjusted_alpha': self.risk_adjusted_alpha,
            'downside_capture_ratio': self.downside_capture_ratio,
            'upside_capture_ratio': self.upside_capture_ratio,
            'capture_ratio': self.capture_ratio
        }


class AdvancedMetricsCalculator:
    """
    Advanced metrics calculator with VaR/CVaR integration, bootstrap confidence intervals,
    and streaming calculations for real-time risk monitoring.
    """
    
    def __init__(
        self,
        var_calculator=None,
        bootstrap_samples: int = 1000,
        confidence_levels: List[float] = [0.90, 0.95, 0.99],
        block_size: int = 50,
        max_workers: int = 4
    ):
        """
        Initialize advanced metrics calculator
        
        Args:
            var_calculator: VaR calculator instance (optional)
            bootstrap_samples: Number of bootstrap samples for confidence intervals
            confidence_levels: Confidence levels for bootstrap intervals
            block_size: Block size for block bootstrap
            max_workers: Maximum number of worker threads
        """
        self.var_calculator = var_calculator
        self.bootstrap_samples = bootstrap_samples
        self.confidence_levels = confidence_levels
        self.block_size = block_size
        self.max_workers = max_workers
        
        # Streaming metrics storage
        self.streaming_metrics: Dict[str, StreamingMetric] = {}
        
        # Cache for expensive computations
        self._cache = {}
        
        logger.info("AdvancedMetricsCalculator initialized",
                   extra={
                       'bootstrap_samples': bootstrap_samples,
                       'confidence_levels': confidence_levels,
                       'block_size': block_size,
                       'max_workers': max_workers
                   })
    
    @njit
    def _calculate_var_historical(self, returns: np.ndarray, confidence_level: float) -> float:
        """Calculate historical VaR (JIT optimized)"""
        if len(returns) == 0:
            return 0.0
        
        # Sort returns and find percentile
        sorted_returns = np.sort(returns)
        index = int((1 - confidence_level) * len(sorted_returns))
        
        if index >= len(sorted_returns):
            index = len(sorted_returns) - 1
        
        return abs(sorted_returns[index])
    
    @njit
    def _calculate_cvar_historical(self, returns: np.ndarray, confidence_level: float) -> float:
        """Calculate historical CVaR (Expected Shortfall) - JIT optimized"""
        if len(returns) == 0:
            return 0.0
        
        # Sort returns and find VaR threshold
        sorted_returns = np.sort(returns)
        var_index = int((1 - confidence_level) * len(sorted_returns))
        
        if var_index >= len(sorted_returns):
            var_index = len(sorted_returns) - 1
        
        # CVaR is the mean of returns beyond VaR
        tail_returns = sorted_returns[:var_index + 1]
        
        if len(tail_returns) == 0:
            return 0.0
        
        return abs(np.mean(tail_returns))
    
    @njit
    def _calculate_parametric_var(
        self, 
        returns: np.ndarray, 
        confidence_level: float
    ) -> float:
        """Calculate parametric VaR assuming normal distribution"""
        if len(returns) == 0:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Z-score for confidence level
        z_score = 0.0
        if confidence_level == 0.95:
            z_score = 1.645
        elif confidence_level == 0.99:
            z_score = 2.326
        elif confidence_level == 0.90:
            z_score = 1.282
        else:
            # Approximate for other confidence levels
            z_score = 1.645  # Default to 95%
        
        var = abs(mean_return - z_score * std_return)
        return var
    
    def calculate_var_cvar_metrics(
        self,
        returns: np.ndarray,
        confidence_level: float = 0.95,
        method: str = "historical"
    ) -> Tuple[float, float]:
        """
        Calculate VaR and CVaR metrics
        
        Args:
            returns: Array of returns
            confidence_level: Confidence level (0.95 = 95%)
            method: Calculation method ('historical', 'parametric', 'external')
        
        Returns:
            Tuple of (VaR, CVaR)
        """
        if len(returns) == 0:
            return 0.0, 0.0
        
        if method == "external" and self.var_calculator is not None:
            # Use external VaR calculator if available
            try:
                # This would integrate with the existing VaR system
                var_result = asyncio.run(self.var_calculator.calculate_var(
                    confidence_level=confidence_level,
                    time_horizon=1,
                    method="parametric"
                ))
                
                if var_result:
                    portfolio_var = var_result.portfolio_var
                    # Calculate CVaR using the VaR result
                    cvar = self._calculate_cvar_historical(returns, confidence_level)
                    return portfolio_var, cvar
                    
            except Exception as e:
                logger.warning(f"External VaR calculation failed: {e}")
                # Fallback to historical method
        
        if method == "parametric":
            var = self._calculate_parametric_var(returns, confidence_level)
            cvar = self._calculate_cvar_historical(returns, confidence_level)
        else:
            # Default to historical method
            var = self._calculate_var_historical(returns, confidence_level)
            cvar = self._calculate_cvar_historical(returns, confidence_level)
        
        return var, cvar
    
    def calculate_risk_adjusted_metrics(
        self,
        returns: np.ndarray,
        benchmark_returns: Optional[np.ndarray] = None,
        risk_free_rate: float = 0.0,
        confidence_level: float = 0.95,
        periods_per_year: int = 252
    ) -> RiskAdjustedMetrics:
        """
        Calculate comprehensive risk-adjusted metrics
        
        Args:
            returns: Array of strategy returns
            benchmark_returns: Optional benchmark returns
            risk_free_rate: Annual risk-free rate
            confidence_level: Confidence level for VaR/CVaR
            periods_per_year: Number of periods per year
        
        Returns:
            RiskAdjustedMetrics object
        """
        if len(returns) == 0:
            return RiskAdjustedMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        # Calculate VaR and CVaR
        var, cvar = self.calculate_var_cvar_metrics(returns, confidence_level)
        
        # Basic risk-adjusted metrics
        excess_returns = returns - risk_free_rate / periods_per_year
        annual_return = np.mean(returns) * periods_per_year
        annual_excess_return = np.mean(excess_returns) * periods_per_year
        
        # Sharpe ratio
        sharpe_ratio = 0.0
        if np.std(excess_returns) > 0:
            sharpe_ratio = np.sqrt(periods_per_year) * np.mean(excess_returns) / np.std(excess_returns)
        
        # Sortino ratio
        downside_returns = excess_returns[excess_returns < 0]
        sortino_ratio = 0.0
        if len(downside_returns) > 0 and np.std(downside_returns) > 0:
            sortino_ratio = np.sqrt(periods_per_year) * np.mean(excess_returns) / np.std(downside_returns)
        
        # Calmar ratio (placeholder - needs max drawdown calculation)
        calmar_ratio = 0.0
        
        # VaR-adjusted return
        var_adjusted_return = annual_excess_return / var if var > 0 else 0.0
        
        # CVaR-adjusted return
        cvar_adjusted_return = annual_excess_return / cvar if cvar > 0 else 0.0
        
        # Risk-adjusted alpha (simplified)
        risk_adjusted_alpha = annual_excess_return
        
        # Capture ratios
        downside_capture_ratio = 0.0
        upside_capture_ratio = 0.0
        capture_ratio = 0.0
        
        if benchmark_returns is not None and len(benchmark_returns) == len(returns):
            # Calculate capture ratios
            upside_benchmark = benchmark_returns[benchmark_returns > 0]
            downside_benchmark = benchmark_returns[benchmark_returns < 0]
            upside_strategy = returns[benchmark_returns > 0]
            downside_strategy = returns[benchmark_returns < 0]
            
            if len(upside_benchmark) > 0 and np.mean(upside_benchmark) > 0:
                upside_capture_ratio = np.mean(upside_strategy) / np.mean(upside_benchmark)
            
            if len(downside_benchmark) > 0 and np.mean(downside_benchmark) < 0:
                downside_capture_ratio = np.mean(downside_strategy) / np.mean(downside_benchmark)
            
            if downside_capture_ratio != 0:
                capture_ratio = upside_capture_ratio / downside_capture_ratio
        
        return RiskAdjustedMetrics(
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            var_adjusted_return=var_adjusted_return,
            cvar_adjusted_return=cvar_adjusted_return,
            risk_adjusted_alpha=risk_adjusted_alpha,
            downside_capture_ratio=downside_capture_ratio,
            upside_capture_ratio=upside_capture_ratio,
            capture_ratio=capture_ratio
        )
    
    def _block_bootstrap_sample(self, returns: np.ndarray, block_size: int) -> np.ndarray:
        """Generate block bootstrap sample"""
        n = len(returns)
        if n <= block_size:
            return returns
        
        # Number of blocks needed
        n_blocks = (n + block_size - 1) // block_size
        
        # Generate random block starting positions
        block_starts = np.random.randint(0, n - block_size + 1, n_blocks)
        
        # Create bootstrap sample
        bootstrap_sample = []
        for start in block_starts:
            block = returns[start:start + block_size]
            bootstrap_sample.extend(block)
        
        return np.array(bootstrap_sample[:n])
    
    def _calculate_metric_bootstrap(
        self,
        returns: np.ndarray,
        metric_func: callable,
        n_samples: int = 1000,
        **kwargs
    ) -> np.ndarray:
        """Calculate bootstrap samples for a metric"""
        bootstrap_values = []
        
        for _ in range(n_samples):
            # Generate bootstrap sample
            bootstrap_returns = self._block_bootstrap_sample(returns, self.block_size)
            
            # Calculate metric on bootstrap sample
            try:
                metric_value = metric_func(bootstrap_returns, **kwargs)
                bootstrap_values.append(metric_value)
            except Exception as e:
                logger.warning(f"Bootstrap calculation failed: {e}")
                continue
        
        return np.array(bootstrap_values)
    
    def calculate_metric_with_confidence(
        self,
        returns: np.ndarray,
        metric_func: callable,
        confidence_level: float = 0.95,
        **kwargs
    ) -> MetricWithConfidence:
        """
        Calculate metric with bootstrap confidence interval
        
        Args:
            returns: Array of returns
            metric_func: Function to calculate metric
            confidence_level: Confidence level for interval
            **kwargs: Additional arguments for metric function
        
        Returns:
            MetricWithConfidence object
        """
        # Calculate actual metric value
        actual_value = metric_func(returns, **kwargs)
        
        # Calculate bootstrap samples
        bootstrap_samples = self._calculate_metric_bootstrap(
            returns, metric_func, self.bootstrap_samples, **kwargs
        )
        
        if len(bootstrap_samples) == 0:
            # Fallback if bootstrap fails
            confidence_interval = ConfidenceInterval(
                lower=actual_value,
                upper=actual_value,
                confidence_level=confidence_level
            )
        else:
            # Calculate confidence interval
            alpha = 1 - confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            lower_bound = np.percentile(bootstrap_samples, lower_percentile)
            upper_bound = np.percentile(bootstrap_samples, upper_percentile)
            
            confidence_interval = ConfidenceInterval(
                lower=lower_bound,
                upper=upper_bound,
                confidence_level=confidence_level
            )
        
        return MetricWithConfidence(
            value=actual_value,
            confidence_interval=confidence_interval,
            bootstrap_samples=len(bootstrap_samples)
        )
    
    def calculate_parallel_metrics(
        self,
        returns: np.ndarray,
        metric_functions: Dict[str, callable],
        confidence_level: float = 0.95,
        **kwargs
    ) -> Dict[str, MetricWithConfidence]:
        """
        Calculate multiple metrics in parallel with confidence intervals
        
        Args:
            returns: Array of returns
            metric_functions: Dictionary of metric name -> function
            confidence_level: Confidence level for intervals
            **kwargs: Additional arguments for metric functions
        
        Returns:
            Dictionary of metric name -> MetricWithConfidence
        """
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_metric = {
                executor.submit(
                    self.calculate_metric_with_confidence,
                    returns,
                    func,
                    confidence_level,
                    **kwargs
                ): metric_name
                for metric_name, func in metric_functions.items()
            }
            
            # Collect results
            for future in as_completed(future_to_metric):
                metric_name = future_to_metric[future]
                try:
                    result = future.result()
                    results[metric_name] = result
                except Exception as e:
                    logger.error(f"Parallel metric calculation failed for {metric_name}: {e}")
                    # Create dummy result
                    results[metric_name] = MetricWithConfidence(
                        value=0.0,
                        confidence_interval=ConfidenceInterval(0.0, 0.0, confidence_level),
                        bootstrap_samples=0
                    )
        
        return results
    
    def update_streaming_metric(
        self,
        metric_name: str,
        new_value: float
    ) -> None:
        """
        Update streaming metric with new value
        
        Args:
            metric_name: Name of the metric
            new_value: New value to incorporate
        """
        if metric_name not in self.streaming_metrics:
            self.streaming_metrics[metric_name] = StreamingMetric(
                value=new_value,
                count=1,
                last_update=datetime.now()
            )
        else:
            self.streaming_metrics[metric_name].update(new_value)
    
    def get_streaming_metrics(self) -> Dict[str, StreamingMetric]:
        """Get all streaming metrics"""
        return self.streaming_metrics.copy()
    
    def reset_streaming_metrics(self) -> None:
        """Reset all streaming metrics"""
        self.streaming_metrics.clear()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary of the calculator"""
        return {
            'bootstrap_samples': self.bootstrap_samples,
            'confidence_levels': self.confidence_levels,
            'block_size': self.block_size,
            'max_workers': self.max_workers,
            'streaming_metrics_count': len(self.streaming_metrics),
            'cache_size': len(self._cache),
            'var_calculator_available': self.var_calculator is not None
        }


# Convenience functions for common metrics
def calculate_sharpe_with_confidence(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    confidence_level: float = 0.95,
    bootstrap_samples: int = 1000
) -> MetricWithConfidence:
    """Calculate Sharpe ratio with confidence interval"""
    
    def sharpe_func(rets, rf_rate, periods):
        if len(rets) == 0:
            return 0.0
        excess_rets = rets - rf_rate / periods
        if np.std(excess_rets) == 0:
            return 0.0
        return np.sqrt(periods) * np.mean(excess_rets) / np.std(excess_rets)
    
    calculator = AdvancedMetricsCalculator(bootstrap_samples=bootstrap_samples)
    return calculator.calculate_metric_with_confidence(
        returns,
        sharpe_func,
        confidence_level,
        rf_rate=risk_free_rate,
        periods=periods_per_year
    )


def calculate_sortino_with_confidence(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    confidence_level: float = 0.95,
    bootstrap_samples: int = 1000
) -> MetricWithConfidence:
    """Calculate Sortino ratio with confidence interval"""
    
    def sortino_func(rets, rf_rate, periods):
        if len(rets) == 0:
            return 0.0
        excess_rets = rets - rf_rate / periods
        downside_rets = excess_rets[excess_rets < 0]
        if len(downside_rets) == 0:
            return float('inf')
        downside_dev = np.std(downside_rets)
        if downside_dev == 0:
            return 0.0
        return np.sqrt(periods) * np.mean(excess_rets) / downside_dev
    
    calculator = AdvancedMetricsCalculator(bootstrap_samples=bootstrap_samples)
    return calculator.calculate_metric_with_confidence(
        returns,
        sortino_func,
        confidence_level,
        rf_rate=risk_free_rate,
        periods=periods_per_year
    )