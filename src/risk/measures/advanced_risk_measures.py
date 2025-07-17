"""
Advanced Risk Measures Module
=============================

This module implements advanced risk measures including:
- Value at Risk (VaR) calculations with multiple methods
- Expected Shortfall (ES) and tail risk metrics
- Maximum drawdown protection
- Risk-adjusted return optimization
- Correlation risk management
- Stress testing and scenario analysis

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
from numba import jit, njit, prange
from scipy import stats, optimize
from scipy.stats import norm, t, genpareto
from sklearn.mixture import GaussianMixture
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class VaRMethod(Enum):
    """VaR calculation methods"""
    HISTORICAL = "historical"
    PARAMETRIC = "parametric"
    MONTE_CARLO = "monte_carlo"
    EXTREME_VALUE = "extreme_value"
    CORNISH_FISHER = "cornish_fisher"


class TailRiskMethod(Enum):
    """Tail risk calculation methods"""
    EXPECTED_SHORTFALL = "expected_shortfall"
    CONDITIONAL_VAR = "conditional_var"
    TAIL_EXPECTATION = "tail_expectation"
    EXTREME_VALUE_THEORY = "extreme_value_theory"


@dataclass
class RiskMeasureConfig:
    """Configuration for risk measures"""
    # VaR parameters
    confidence_levels: List[float] = field(default_factory=lambda: [0.95, 0.99, 0.999])
    time_horizons: List[int] = field(default_factory=lambda: [1, 5, 10, 22])
    lookback_period: int = 252
    
    # Monte Carlo parameters
    num_simulations: int = 10000
    random_seed: int = 42
    
    # Extreme Value Theory parameters
    threshold_percentile: float = 0.95
    block_size: int = 22
    
    # Cornish-Fisher parameters
    use_higher_moments: bool = True
    
    # Stress testing parameters
    stress_scenarios: List[str] = field(default_factory=lambda: [
        "market_crash", "volatility_spike", "correlation_breakdown", "liquidity_crisis"
    ])
    
    # Performance parameters
    parallel_processing: bool = True
    max_workers: int = 4
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'confidence_levels': self.confidence_levels,
            'time_horizons': self.time_horizons,
            'lookback_period': self.lookback_period,
            'num_simulations': self.num_simulations,
            'random_seed': self.random_seed,
            'threshold_percentile': self.threshold_percentile,
            'block_size': self.block_size,
            'use_higher_moments': self.use_higher_moments,
            'stress_scenarios': self.stress_scenarios,
            'parallel_processing': self.parallel_processing,
            'max_workers': self.max_workers
        }


@dataclass
class VaRResult:
    """VaR calculation result"""
    method: str
    confidence_level: float
    time_horizon: int
    var_value: float
    expected_shortfall: float
    tail_expectation: float
    confidence_interval: Tuple[float, float]
    model_parameters: Dict[str, Any]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'method': self.method,
            'confidence_level': self.confidence_level,
            'time_horizon': self.time_horizon,
            'var_value': self.var_value,
            'expected_shortfall': self.expected_shortfall,
            'tail_expectation': self.tail_expectation,
            'confidence_interval': self.confidence_interval,
            'model_parameters': self.model_parameters,
            'metadata': self.metadata
        }


@dataclass
class TailRiskResult:
    """Tail risk calculation result"""
    method: str
    confidence_level: float
    expected_shortfall: float
    tail_expectation: float
    tail_probability: float
    extreme_quantiles: Dict[str, float]
    tail_distribution: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'method': self.method,
            'confidence_level': self.confidence_level,
            'expected_shortfall': self.expected_shortfall,
            'tail_expectation': self.tail_expectation,
            'tail_probability': self.tail_probability,
            'extreme_quantiles': self.extreme_quantiles,
            'tail_distribution': self.tail_distribution
        }


@dataclass
class StressTestResult:
    """Stress test result"""
    scenario_name: str
    portfolio_loss: float
    individual_losses: Dict[str, float]
    risk_contribution: Dict[str, float]
    correlation_impact: float
    liquidity_impact: float
    recovery_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'scenario_name': self.scenario_name,
            'portfolio_loss': self.portfolio_loss,
            'individual_losses': self.individual_losses,
            'risk_contribution': self.risk_contribution,
            'correlation_impact': self.correlation_impact,
            'liquidity_impact': self.liquidity_impact,
            'recovery_time': self.recovery_time
        }


@dataclass
class DrawdownAnalysis:
    """Drawdown analysis result"""
    max_drawdown: float
    max_drawdown_duration: int
    current_drawdown: float
    drawdown_probability: float
    expected_recovery_time: float
    underwater_periods: List[Tuple[datetime, datetime]]
    drawdown_distribution: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'max_drawdown': self.max_drawdown,
            'max_drawdown_duration': self.max_drawdown_duration,
            'current_drawdown': self.current_drawdown,
            'drawdown_probability': self.drawdown_probability,
            'expected_recovery_time': self.expected_recovery_time,
            'underwater_periods': [(start.isoformat(), end.isoformat()) for start, end in self.underwater_periods],
            'drawdown_distribution': self.drawdown_distribution
        }


# Numba JIT optimized functions
@njit
def calculate_historical_var(
    returns: np.ndarray,
    confidence_level: float,
    time_horizon: int = 1
) -> float:
    """Calculate historical VaR - JIT optimized"""
    
    if len(returns) == 0:
        return 0.0
    
    # Scale returns for time horizon
    scaled_returns = returns * np.sqrt(time_horizon)
    
    # Sort returns
    sorted_returns = np.sort(scaled_returns)
    
    # Calculate VaR index
    var_index = int((1 - confidence_level) * len(sorted_returns))
    
    if var_index >= len(sorted_returns):
        var_index = len(sorted_returns) - 1
    
    return abs(sorted_returns[var_index])


@njit
def calculate_parametric_var(
    returns: np.ndarray,
    confidence_level: float,
    time_horizon: int = 1
) -> float:
    """Calculate parametric VaR assuming normal distribution - JIT optimized"""
    
    if len(returns) == 0:
        return 0.0
    
    # Calculate mean and standard deviation
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    # Scale for time horizon
    scaled_mean = mean_return * time_horizon
    scaled_std = std_return * np.sqrt(time_horizon)
    
    # Calculate VaR using normal distribution
    z_score = stats.norm.ppf(1 - confidence_level)
    var_value = scaled_mean + z_score * scaled_std
    
    return abs(var_value)


@njit
def calculate_expected_shortfall_jit(
    returns: np.ndarray,
    confidence_level: float,
    time_horizon: int = 1
) -> float:
    """Calculate Expected Shortfall - JIT optimized"""
    
    if len(returns) == 0:
        return 0.0
    
    # Scale returns for time horizon
    scaled_returns = returns * np.sqrt(time_horizon)
    
    # Sort returns
    sorted_returns = np.sort(scaled_returns)
    
    # Find VaR threshold
    var_index = int((1 - confidence_level) * len(sorted_returns))
    
    if var_index >= len(sorted_returns):
        var_index = len(sorted_returns) - 1
    
    # Expected Shortfall is mean of tail beyond VaR
    if var_index == 0:
        return abs(sorted_returns[0])
    
    tail_returns = sorted_returns[:var_index]
    
    if len(tail_returns) == 0:
        return 0.0
    
    return abs(np.mean(tail_returns))


@njit
def calculate_maximum_drawdown_jit(equity_curve: np.ndarray) -> Tuple[float, int]:
    """Calculate maximum drawdown and duration - JIT optimized"""
    
    if len(equity_curve) == 0:
        return 0.0, 0
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(equity_curve)
    
    # Calculate drawdown
    drawdown = (equity_curve - running_max) / running_max
    
    # Find maximum drawdown
    max_drawdown = np.min(drawdown)
    max_drawdown_idx = np.argmin(drawdown)
    
    # Calculate duration
    duration = 0
    for i in range(max_drawdown_idx, len(drawdown)):
        if drawdown[i] >= 0:
            duration = i - max_drawdown_idx
            break
    else:
        duration = len(drawdown) - max_drawdown_idx
    
    return abs(max_drawdown), duration


@njit
def calculate_tail_expectation_jit(
    returns: np.ndarray,
    threshold: float
) -> float:
    """Calculate tail expectation beyond threshold - JIT optimized"""
    
    if len(returns) == 0:
        return 0.0
    
    # Get tail returns
    tail_returns = returns[returns <= threshold]
    
    if len(tail_returns) == 0:
        return 0.0
    
    return abs(np.mean(tail_returns))


@njit
def calculate_cornish_fisher_var(
    returns: np.ndarray,
    confidence_level: float,
    time_horizon: int = 1
) -> float:
    """Calculate Cornish-Fisher VaR with skewness and kurtosis - JIT optimized"""
    
    if len(returns) == 0:
        return 0.0
    
    # Calculate moments
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    # Calculate skewness and kurtosis
    skewness = np.mean(((returns - mean_return) / std_return) ** 3)
    kurtosis = np.mean(((returns - mean_return) / std_return) ** 4) - 3
    
    # Get normal quantile
    z = stats.norm.ppf(1 - confidence_level)
    
    # Cornish-Fisher expansion
    cf_quantile = (z + 
                   (z**2 - 1) * skewness / 6 + 
                   (z**3 - 3*z) * kurtosis / 24 - 
                   (2*z**3 - 5*z) * skewness**2 / 36)
    
    # Scale for time horizon
    scaled_mean = mean_return * time_horizon
    scaled_std = std_return * np.sqrt(time_horizon)
    
    var_value = scaled_mean + cf_quantile * scaled_std
    
    return abs(var_value)


@njit(parallel=True)
def monte_carlo_simulation(
    mean_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    num_simulations: int,
    time_horizon: int,
    random_seed: int = 42
) -> np.ndarray:
    """Monte Carlo simulation for portfolio returns - JIT optimized"""
    
    n_assets = len(mean_returns)
    
    # Set random seed
    np.random.seed(random_seed)
    
    # Generate random samples
    portfolio_returns = np.zeros(num_simulations)
    
    for i in prange(num_simulations):
        # Generate random normal vector
        random_vector = np.random.randn(n_assets)
        
        # Apply Cholesky decomposition
        cholesky = np.linalg.cholesky(covariance_matrix)
        correlated_returns = np.dot(cholesky, random_vector)
        
        # Scale for time horizon
        scaled_returns = mean_returns * time_horizon + correlated_returns * np.sqrt(time_horizon)
        
        # Sum to get portfolio return
        portfolio_returns[i] = np.sum(scaled_returns)
    
    return portfolio_returns


class AdvancedRiskMeasures:
    """
    Advanced Risk Measures Calculator
    
    This class implements sophisticated risk measurement techniques including
    multiple VaR methods, tail risk analysis, stress testing, and drawdown analysis.
    """
    
    def __init__(self, config: RiskMeasureConfig):
        """
        Initialize the advanced risk measures calculator
        
        Args:
            config: Risk measure configuration
        """
        self.config = config
        
        # Results storage
        self.var_results: Dict[str, VaRResult] = {}
        self.tail_risk_results: Dict[str, TailRiskResult] = {}
        self.stress_test_results: Dict[str, StressTestResult] = {}
        self.drawdown_analyses: Dict[str, DrawdownAnalysis] = {}
        
        # Performance tracking
        self.calculation_times: Dict[str, List[float]] = {}
        
        # Stress scenarios
        self.stress_scenarios = self._initialize_stress_scenarios()
        
        logger.info("AdvancedRiskMeasures initialized",
                   extra={'config': config.to_dict()})
    
    def _initialize_stress_scenarios(self) -> Dict[str, Dict[str, Any]]:
        """Initialize stress test scenarios"""
        
        scenarios = {
            'market_crash': {
                'description': 'Market crash scenario (-20% market drop)',
                'market_shock': -0.20,
                'volatility_multiplier': 3.0,
                'correlation_increase': 0.30,
                'liquidity_impact': 0.15
            },
            'volatility_spike': {
                'description': 'Volatility spike scenario (300% vol increase)',
                'market_shock': 0.0,
                'volatility_multiplier': 4.0,
                'correlation_increase': 0.20,
                'liquidity_impact': 0.10
            },
            'correlation_breakdown': {
                'description': 'Correlation breakdown scenario',
                'market_shock': -0.10,
                'volatility_multiplier': 2.0,
                'correlation_increase': 0.50,
                'liquidity_impact': 0.20
            },
            'liquidity_crisis': {
                'description': 'Liquidity crisis scenario',
                'market_shock': -0.15,
                'volatility_multiplier': 2.5,
                'correlation_increase': 0.40,
                'liquidity_impact': 0.30
            }
        }
        
        return scenarios
    
    async def calculate_var(
        self,
        returns: np.ndarray,
        method: VaRMethod = VaRMethod.HISTORICAL,
        confidence_level: float = 0.95,
        time_horizon: int = 1,
        weights: Optional[np.ndarray] = None
    ) -> VaRResult:
        """
        Calculate Value at Risk using specified method
        
        Args:
            returns: Return series or matrix
            method: VaR calculation method
            confidence_level: Confidence level
            time_horizon: Time horizon in days
            weights: Portfolio weights (for multivariate returns)
        
        Returns:
            VaRResult object
        """
        start_time = datetime.now()
        
        try:
            # Prepare returns
            if returns.ndim == 1:
                portfolio_returns = returns
            else:
                # Convert to portfolio returns
                if weights is None:
                    weights = np.ones(returns.shape[1]) / returns.shape[1]
                portfolio_returns = np.dot(returns, weights)
            
            # Calculate VaR based on method
            if method == VaRMethod.HISTORICAL:
                var_value = calculate_historical_var(portfolio_returns, confidence_level, time_horizon)
                model_parameters = {'method': 'historical'}
                
            elif method == VaRMethod.PARAMETRIC:
                var_value = calculate_parametric_var(portfolio_returns, confidence_level, time_horizon)
                model_parameters = {
                    'method': 'parametric',
                    'mean': np.mean(portfolio_returns),
                    'std': np.std(portfolio_returns)
                }
                
            elif method == VaRMethod.MONTE_CARLO:
                var_value = await self._calculate_monte_carlo_var(
                    portfolio_returns, confidence_level, time_horizon
                )
                model_parameters = {
                    'method': 'monte_carlo',
                    'num_simulations': self.config.num_simulations
                }
                
            elif method == VaRMethod.EXTREME_VALUE:
                var_value = await self._calculate_extreme_value_var(
                    portfolio_returns, confidence_level, time_horizon
                )
                model_parameters = {'method': 'extreme_value'}
                
            elif method == VaRMethod.CORNISH_FISHER:
                var_value = calculate_cornish_fisher_var(portfolio_returns, confidence_level, time_horizon)
                model_parameters = {'method': 'cornish_fisher'}
                
            else:
                raise ValueError(f"Unknown VaR method: {method}")
            
            # Calculate Expected Shortfall
            expected_shortfall = calculate_expected_shortfall_jit(
                portfolio_returns, confidence_level, time_horizon
            )
            
            # Calculate tail expectation
            var_threshold = -var_value
            tail_expectation = calculate_tail_expectation_jit(portfolio_returns, var_threshold)
            
            # Calculate confidence interval (bootstrap)
            confidence_interval = await self._calculate_var_confidence_interval(
                portfolio_returns, method, confidence_level, time_horizon
            )
            
            # Create result
            result = VaRResult(
                method=method.value,
                confidence_level=confidence_level,
                time_horizon=time_horizon,
                var_value=var_value,
                expected_shortfall=expected_shortfall,
                tail_expectation=tail_expectation,
                confidence_interval=confidence_interval,
                model_parameters=model_parameters,
                metadata={
                    'calculation_time': (datetime.now() - start_time).total_seconds(),
                    'data_points': len(portfolio_returns)
                }
            )
            
            # Store result
            key = f"{method.value}_{confidence_level}_{time_horizon}"
            self.var_results[key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"VaR calculation failed: {e}")
            raise
        
        finally:
            # Track performance
            calc_time = (datetime.now() - start_time).total_seconds() * 1000
            method_name = method.value if hasattr(method, 'value') else str(method)
            
            if method_name not in self.calculation_times:
                self.calculation_times[method_name] = []
            
            self.calculation_times[method_name].append(calc_time)
            
            # Keep only recent times
            if len(self.calculation_times[method_name]) > 1000:
                self.calculation_times[method_name] = self.calculation_times[method_name][-1000:]
    
    async def _calculate_monte_carlo_var(
        self,
        returns: np.ndarray,
        confidence_level: float,
        time_horizon: int
    ) -> float:
        """Calculate Monte Carlo VaR"""
        
        # Estimate parameters
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Create single-asset covariance matrix
        covariance_matrix = np.array([[std_return**2]])
        mean_returns = np.array([mean_return])
        
        # Run simulation
        simulated_returns = monte_carlo_simulation(
            mean_returns,
            covariance_matrix,
            self.config.num_simulations,
            time_horizon,
            self.config.random_seed
        )
        
        # Calculate VaR from simulated returns
        var_value = calculate_historical_var(simulated_returns, confidence_level, 1)
        
        return var_value
    
    async def _calculate_extreme_value_var(
        self,
        returns: np.ndarray,
        confidence_level: float,
        time_horizon: int
    ) -> float:
        """Calculate VaR using Extreme Value Theory"""
        
        # Get threshold
        threshold = np.percentile(returns, (1 - self.config.threshold_percentile) * 100)
        
        # Get exceedances
        exceedances = returns[returns <= threshold] - threshold
        
        if len(exceedances) == 0:
            return calculate_historical_var(returns, confidence_level, time_horizon)
        
        # Fit Generalized Pareto Distribution
        try:
            # Fit GPD parameters
            params = stats.genpareto.fit(exceedances, floc=0)
            
            # Calculate VaR using GPD
            n = len(returns)
            n_exceedances = len(exceedances)
            
            # Probability of exceedance
            prob_exceedance = n_exceedances / n
            
            # GPD quantile
            p = (1 - confidence_level) / prob_exceedance
            
            if p > 1:
                p = 0.99
            
            gpd_quantile = stats.genpareto.ppf(p, *params)
            
            # Scale for time horizon
            var_value = abs(threshold + gpd_quantile) * np.sqrt(time_horizon)
            
            return var_value
            
        except Exception as e:
            logger.warning(f"GPD fitting failed: {e}, using historical VaR")
            return calculate_historical_var(returns, confidence_level, time_horizon)
    
    async def _calculate_var_confidence_interval(
        self,
        returns: np.ndarray,
        method: VaRMethod,
        confidence_level: float,
        time_horizon: int,
        n_bootstrap: int = 1000
    ) -> Tuple[float, float]:
        """Calculate VaR confidence interval using bootstrap"""
        
        bootstrap_vars = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            bootstrap_sample = np.random.choice(returns, size=len(returns), replace=True)
            
            # Calculate VaR
            if method == VaRMethod.HISTORICAL:
                var_value = calculate_historical_var(bootstrap_sample, confidence_level, time_horizon)
            elif method == VaRMethod.PARAMETRIC:
                var_value = calculate_parametric_var(bootstrap_sample, confidence_level, time_horizon)
            else:
                var_value = calculate_historical_var(bootstrap_sample, confidence_level, time_horizon)
            
            bootstrap_vars.append(var_value)
        
        # Calculate confidence interval
        lower_bound = np.percentile(bootstrap_vars, 2.5)
        upper_bound = np.percentile(bootstrap_vars, 97.5)
        
        return (lower_bound, upper_bound)
    
    async def calculate_tail_risk(
        self,
        returns: np.ndarray,
        method: TailRiskMethod = TailRiskMethod.EXPECTED_SHORTFALL,
        confidence_level: float = 0.95
    ) -> TailRiskResult:
        """
        Calculate tail risk measures
        
        Args:
            returns: Return series
            method: Tail risk calculation method
            confidence_level: Confidence level
        
        Returns:
            TailRiskResult object
        """
        
        if method == TailRiskMethod.EXPECTED_SHORTFALL:
            expected_shortfall = calculate_expected_shortfall_jit(returns, confidence_level)
            
            # Calculate tail expectation
            var_threshold = -calculate_historical_var(returns, confidence_level)
            tail_expectation = calculate_tail_expectation_jit(returns, var_threshold)
            
            # Calculate tail probability
            tail_returns = returns[returns <= var_threshold]
            tail_probability = len(tail_returns) / len(returns)
            
            # Calculate extreme quantiles
            extreme_quantiles = {
                'q_99': np.percentile(returns, 1),
                'q_999': np.percentile(returns, 0.1),
                'q_9999': np.percentile(returns, 0.01)
            }
            
            result = TailRiskResult(
                method=method.value,
                confidence_level=confidence_level,
                expected_shortfall=expected_shortfall,
                tail_expectation=tail_expectation,
                tail_probability=tail_probability,
                extreme_quantiles=extreme_quantiles,
                tail_distribution={'method': 'empirical'}
            )
            
        else:
            raise ValueError(f"Unknown tail risk method: {method}")
        
        # Store result
        key = f"{method.value}_{confidence_level}"
        self.tail_risk_results[key] = result
        
        return result
    
    async def calculate_maximum_drawdown(
        self,
        equity_curve: np.ndarray,
        returns: Optional[np.ndarray] = None
    ) -> DrawdownAnalysis:
        """
        Calculate comprehensive drawdown analysis
        
        Args:
            equity_curve: Portfolio equity curve
            returns: Optional return series
        
        Returns:
            DrawdownAnalysis object
        """
        
        # Calculate maximum drawdown
        max_drawdown, max_drawdown_duration = calculate_maximum_drawdown_jit(equity_curve)
        
        # Calculate current drawdown
        current_value = equity_curve[-1]
        peak_value = np.max(equity_curve)
        current_drawdown = (peak_value - current_value) / peak_value if peak_value > 0 else 0.0
        
        # Calculate drawdown probability (simplified)
        drawdown_probability = self._estimate_drawdown_probability(equity_curve, max_drawdown)
        
        # Estimate recovery time
        expected_recovery_time = self._estimate_recovery_time(equity_curve, returns)
        
        # Find underwater periods
        underwater_periods = self._find_underwater_periods(equity_curve)
        
        # Calculate drawdown distribution
        drawdown_distribution = self._calculate_drawdown_distribution(equity_curve)
        
        result = DrawdownAnalysis(
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_drawdown_duration,
            current_drawdown=current_drawdown,
            drawdown_probability=drawdown_probability,
            expected_recovery_time=expected_recovery_time,
            underwater_periods=underwater_periods,
            drawdown_distribution=drawdown_distribution
        )
        
        # Store result
        self.drawdown_analyses['current'] = result
        
        return result
    
    def _estimate_drawdown_probability(self, equity_curve: np.ndarray, threshold: float) -> float:
        """Estimate probability of drawdown exceeding threshold"""
        
        # Calculate rolling drawdowns
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - running_max) / running_max
        
        # Count exceedances
        exceedances = np.sum(drawdowns <= -threshold)
        
        return exceedances / len(drawdowns)
    
    def _estimate_recovery_time(self, equity_curve: np.ndarray, returns: Optional[np.ndarray]) -> float:
        """Estimate expected recovery time from drawdown"""
        
        if returns is None:
            return 0.0
        
        # Calculate average positive return
        positive_returns = returns[returns > 0]
        
        if len(positive_returns) == 0:
            return np.inf
        
        avg_positive_return = np.mean(positive_returns)
        
        # Current drawdown
        current_value = equity_curve[-1]
        peak_value = np.max(equity_curve)
        current_drawdown = (peak_value - current_value) / peak_value
        
        # Estimate recovery time
        if avg_positive_return > 0:
            recovery_time = current_drawdown / avg_positive_return
        else:
            recovery_time = np.inf
        
        return recovery_time
    
    def _find_underwater_periods(self, equity_curve: np.ndarray) -> List[Tuple[datetime, datetime]]:
        """Find periods when portfolio is underwater"""
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(equity_curve)
        
        # Find underwater periods
        underwater = equity_curve < running_max
        
        # Find start and end of underwater periods
        periods = []
        start_idx = None
        
        for i, is_underwater in enumerate(underwater):
            if is_underwater and start_idx is None:
                start_idx = i
            elif not is_underwater and start_idx is not None:
                # End of underwater period
                start_date = datetime.now() - timedelta(days=len(equity_curve) - start_idx)
                end_date = datetime.now() - timedelta(days=len(equity_curve) - i)
                periods.append((start_date, end_date))
                start_idx = None
        
        # Handle case where we're still underwater
        if start_idx is not None:
            start_date = datetime.now() - timedelta(days=len(equity_curve) - start_idx)
            end_date = datetime.now()
            periods.append((start_date, end_date))
        
        return periods
    
    def _calculate_drawdown_distribution(self, equity_curve: np.ndarray) -> Dict[str, float]:
        """Calculate drawdown distribution statistics"""
        
        # Calculate all drawdowns
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - running_max) / running_max
        
        # Calculate statistics
        distribution = {
            'mean': np.mean(drawdowns),
            'std': np.std(drawdowns),
            'min': np.min(drawdowns),
            'max': np.max(drawdowns),
            'q10': np.percentile(drawdowns, 10),
            'q25': np.percentile(drawdowns, 25),
            'q50': np.percentile(drawdowns, 50),
            'q75': np.percentile(drawdowns, 75),
            'q90': np.percentile(drawdowns, 90),
            'q95': np.percentile(drawdowns, 95),
            'q99': np.percentile(drawdowns, 99)
        }
        
        return distribution
    
    async def run_stress_test(
        self,
        returns: np.ndarray,
        weights: np.ndarray,
        scenario_name: str
    ) -> StressTestResult:
        """
        Run stress test scenario
        
        Args:
            returns: Return matrix (time x assets)
            weights: Portfolio weights
            scenario_name: Name of stress scenario
        
        Returns:
            StressTestResult object
        """
        
        if scenario_name not in self.stress_scenarios:
            raise ValueError(f"Unknown stress scenario: {scenario_name}")
        
        scenario = self.stress_scenarios[scenario_name]
        
        # Apply scenario shocks
        stressed_returns = self._apply_stress_scenario(returns, scenario)
        
        # Calculate portfolio loss
        portfolio_returns = np.dot(stressed_returns, weights)
        portfolio_loss = np.sum(portfolio_returns)
        
        # Calculate individual losses
        individual_losses = {}
        for i, asset_returns in enumerate(stressed_returns.T):
            individual_losses[f'asset_{i}'] = np.sum(asset_returns) * weights[i]
        
        # Calculate risk contribution
        risk_contribution = {}
        total_loss = abs(portfolio_loss)
        
        for asset, loss in individual_losses.items():
            if total_loss > 0:
                risk_contribution[asset] = abs(loss) / total_loss
            else:
                risk_contribution[asset] = 0.0
        
        # Calculate correlation and liquidity impacts
        correlation_impact = scenario['correlation_increase']
        liquidity_impact = scenario['liquidity_impact']
        
        # Estimate recovery time (simplified)
        recovery_time = self._estimate_scenario_recovery_time(portfolio_loss, scenario)
        
        result = StressTestResult(
            scenario_name=scenario_name,
            portfolio_loss=portfolio_loss,
            individual_losses=individual_losses,
            risk_contribution=risk_contribution,
            correlation_impact=correlation_impact,
            liquidity_impact=liquidity_impact,
            recovery_time=recovery_time
        )
        
        # Store result
        self.stress_test_results[scenario_name] = result
        
        return result
    
    def _apply_stress_scenario(
        self,
        returns: np.ndarray,
        scenario: Dict[str, Any]
    ) -> np.ndarray:
        """Apply stress scenario to returns"""
        
        stressed_returns = returns.copy()
        
        # Apply market shock
        market_shock = scenario['market_shock']
        stressed_returns += market_shock
        
        # Apply volatility multiplier
        vol_multiplier = scenario['volatility_multiplier']
        mean_returns = np.mean(stressed_returns, axis=0)
        stressed_returns = mean_returns + (stressed_returns - mean_returns) * vol_multiplier
        
        return stressed_returns
    
    def _estimate_scenario_recovery_time(
        self,
        portfolio_loss: float,
        scenario: Dict[str, Any]
    ) -> float:
        """Estimate recovery time from stress scenario"""
        
        # Simple recovery time estimation
        # In practice, this would be more sophisticated
        
        base_recovery = 30  # days
        
        # Adjust based on scenario severity
        if abs(portfolio_loss) > 0.20:
            recovery_time = base_recovery * 3
        elif abs(portfolio_loss) > 0.10:
            recovery_time = base_recovery * 2
        else:
            recovery_time = base_recovery
        
        # Adjust for liquidity impact
        liquidity_impact = scenario['liquidity_impact']
        recovery_time *= (1 + liquidity_impact)
        
        return recovery_time
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary"""
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'var_results': {k: v.to_dict() for k, v in self.var_results.items()},
            'tail_risk_results': {k: v.to_dict() for k, v in self.tail_risk_results.items()},
            'stress_test_results': {k: v.to_dict() for k, v in self.stress_test_results.items()},
            'drawdown_analyses': {k: v.to_dict() for k, v in self.drawdown_analyses.items()},
            'performance_metrics': {
                method: {
                    'avg_calc_time_ms': np.mean(times),
                    'max_calc_time_ms': np.max(times),
                    'calculation_count': len(times)
                }
                for method, times in self.calculation_times.items()
            },
            'config': self.config.to_dict()
        }
        
        return summary


# Factory function
def create_risk_measures(config_dict: Optional[Dict[str, Any]] = None) -> AdvancedRiskMeasures:
    """
    Create advanced risk measures calculator
    
    Args:
        config_dict: Configuration dictionary
    
    Returns:
        AdvancedRiskMeasures instance
    """
    
    if config_dict is None:
        config = RiskMeasureConfig()
    else:
        config = RiskMeasureConfig(**config_dict)
    
    return AdvancedRiskMeasures(config)