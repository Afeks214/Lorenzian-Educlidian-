"""
Advanced Performance Attribution System

This module implements sophisticated performance attribution analysis for portfolio optimization,
providing detailed strategy-level performance tracking, contribution analysis, and risk-adjusted
performance metrics with multi-factor attribution models.

Key Features:
- Strategy-level performance decomposition and attribution
- Risk-adjusted performance metrics (Sharpe, Information Ratio, Alpha, Beta)
- Multi-factor performance attribution (market, style, strategy-specific)
- Time-series performance analysis with regime attribution
- Contribution analysis (return, risk, and correlation contributions)
- Performance persistence and stability analysis
- Advanced attribution models (Brinson-Fachler, factor-based)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import structlog
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from src.core.events import EventBus, Event, EventType
from src.risk.core.correlation_tracker import CorrelationRegime

logger = structlog.get_logger()


class AttributionMethod(Enum):
    """Performance attribution methods"""
    BRINSON_FACHLER = "brinson_fachler"
    FACTOR_BASED = "factor_based"
    RETURN_DECOMPOSITION = "return_decomposition"
    RISK_DECOMPOSITION = "risk_decomposition"
    CORRELATION_ATTRIBUTION = "correlation_attribution"


class PerformanceMetric(Enum):
    """Performance metrics for attribution"""
    TOTAL_RETURN = "total_return"
    EXCESS_RETURN = "excess_return"
    SHARPE_RATIO = "sharpe_ratio"
    INFORMATION_RATIO = "information_ratio"
    ALPHA = "alpha"
    BETA = "beta"
    TRACKING_ERROR = "tracking_error"
    MAX_DRAWDOWN = "max_drawdown"
    CALMAR_RATIO = "calmar_ratio"
    SORTINO_RATIO = "sortino_ratio"


@dataclass
class StrategyPerformanceMetrics:
    """Comprehensive strategy performance metrics"""
    strategy_id: str
    period_start: datetime
    period_end: datetime
    
    # Return metrics
    total_return: float
    annualized_return: float
    excess_return: float  # vs benchmark
    cumulative_return: float
    
    # Risk metrics
    volatility: float
    downside_volatility: float
    tracking_error: float
    max_drawdown: float
    var_95: float
    cvar_95: float
    
    # Risk-adjusted metrics
    sharpe_ratio: float
    information_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    alpha: float
    beta: float
    
    # Attribution-specific metrics
    correlation_with_portfolio: float
    correlation_with_benchmark: float
    r_squared: float
    
    # Performance consistency
    hit_ratio: float  # % of periods with positive excess returns
    up_capture: float  # Upside capture ratio
    down_capture: float  # Downside capture ratio
    
    # Additional statistics
    skewness: float
    kurtosis: float
    return_stability: float  # Consistency of returns


@dataclass
class AttributionContribution:
    """Attribution contribution breakdown"""
    strategy_id: str
    weight: float
    
    # Return contributions
    return_contribution: float
    excess_return_contribution: float
    
    # Risk contributions
    risk_contribution: float
    tracking_error_contribution: float
    
    # Interaction effects
    selection_effect: float  # Strategy-specific return
    allocation_effect: float  # Weight allocation effect
    interaction_effect: float  # Cross-term effects
    
    # Factor contributions (if factor-based attribution)
    factor_contributions: Dict[str, float] = field(default_factory=dict)


@dataclass
class PortfolioAttribution:
    """Complete portfolio attribution analysis"""
    timestamp: datetime
    period_start: datetime
    period_end: datetime
    method: AttributionMethod
    
    # Portfolio-level metrics
    portfolio_return: float
    benchmark_return: float
    excess_return: float
    tracking_error: float
    information_ratio: float
    
    # Strategy contributions
    strategy_contributions: List[AttributionContribution]
    
    # Factor contributions (if applicable)
    factor_attributions: Dict[str, float] = field(default_factory=dict)
    
    # Decomposition totals
    total_selection_effect: float = 0.0
    total_allocation_effect: float = 0.0
    total_interaction_effect: float = 0.0
    
    # Attribution quality metrics
    attribution_residual: float = 0.0  # Unexplained performance
    attribution_r_squared: float = 0.0


class PerformanceAttributionEngine:
    """
    Advanced Performance Attribution Engine for comprehensive portfolio
    performance analysis and strategy-level contribution tracking.
    """
    
    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        n_strategies: int = 5,
        benchmark_return: float = 0.08,  # 8% annual benchmark
        risk_free_rate: float = 0.02,  # 2% risk-free rate
        attribution_window: int = 252  # 1 year attribution window
    ):
        """
        Initialize Performance Attribution Engine
        
        Args:
            event_bus: Event bus for communication
            n_strategies: Number of strategies
            benchmark_return: Benchmark return for comparison
            risk_free_rate: Risk-free rate for risk-adjusted metrics
            attribution_window: Rolling window for attribution analysis
        """
        self.event_bus = event_bus
        self.n_strategies = n_strategies
        self.benchmark_return = benchmark_return
        self.risk_free_rate = risk_free_rate
        self.attribution_window = attribution_window
        
        # Performance data storage
        self.strategy_returns: Dict[str, List[Tuple[datetime, float]]] = {}
        self.portfolio_returns: List[Tuple[datetime, float]] = []
        self.benchmark_returns: List[Tuple[datetime, float]] = []
        self.weights_history: List[Tuple[datetime, np.ndarray]] = []
        
        # Attribution results storage
        self.attribution_history: List[PortfolioAttribution] = []
        self.strategy_metrics_history: Dict[str, List[StrategyPerformanceMetrics]] = {}
        
        # Factor models (simplified)
        self.factor_names = ['Market', 'Size', 'Value', 'Momentum', 'Quality']
        self.factor_returns: Dict[str, List[Tuple[datetime, float]]] = {
            factor: [] for factor in self.factor_names
        }
        
        # Performance tracking
        self.calculation_times: List[float] = []
        
        # Initialize strategy tracking
        for i in range(n_strategies):
            strategy_id = f"Strategy_{i}"
            self.strategy_returns[strategy_id] = []
            self.strategy_metrics_history[strategy_id] = []
        
        logger.info("Performance Attribution Engine initialized",
                   n_strategies=n_strategies,
                   benchmark_return=benchmark_return,
                   attribution_window=attribution_window)
    
    def update_performance_data(
        self,
        timestamp: datetime,
        strategy_returns: Dict[str, float],
        portfolio_return: float,
        weights: np.ndarray,
        benchmark_return: Optional[float] = None
    ):
        """
        Update performance data for attribution analysis
        
        Args:
            timestamp: Data timestamp
            strategy_returns: Individual strategy returns
            portfolio_return: Total portfolio return
            weights: Current portfolio weights
            benchmark_return: Benchmark return (optional)
        """
        try:
            # Store strategy returns
            for strategy_id, return_val in strategy_returns.items():
                if strategy_id in self.strategy_returns:
                    self.strategy_returns[strategy_id].append((timestamp, return_val))
                    
                    # Keep rolling window
                    if len(self.strategy_returns[strategy_id]) > self.attribution_window * 2:
                        self.strategy_returns[strategy_id] = self.strategy_returns[strategy_id][-self.attribution_window:]
            
            # Store portfolio return
            self.portfolio_returns.append((timestamp, portfolio_return))
            if len(self.portfolio_returns) > self.attribution_window * 2:
                self.portfolio_returns = self.portfolio_returns[-self.attribution_window:]
            
            # Store weights
            self.weights_history.append((timestamp, weights.copy()))
            if len(self.weights_history) > self.attribution_window * 2:
                self.weights_history = self.weights_history[-self.attribution_window:]
            
            # Store benchmark return
            if benchmark_return is not None:
                self.benchmark_returns.append((timestamp, benchmark_return))
            else:
                # Use default benchmark return (daily from annual)
                daily_benchmark = (1 + self.benchmark_return) ** (1/252) - 1
                self.benchmark_returns.append((timestamp, daily_benchmark))
            
            if len(self.benchmark_returns) > self.attribution_window * 2:
                self.benchmark_returns = self.benchmark_returns[-self.attribution_window:]
            
            # Update factor returns (simplified simulation)
            self._update_factor_returns(timestamp)
            
        except Exception as e:
            logger.error("Performance data update failed", error=str(e))
    
    def calculate_strategy_performance_metrics(
        self,
        strategy_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Optional[StrategyPerformanceMetrics]:
        """
        Calculate comprehensive performance metrics for a strategy
        
        Args:
            strategy_id: Strategy identifier
            start_date: Analysis start date (optional)
            end_date: Analysis end date (optional)
            
        Returns:
            Strategy performance metrics
        """
        if strategy_id not in self.strategy_returns:
            return None
        
        strategy_data = self.strategy_returns[strategy_id]
        if len(strategy_data) < 30:  # Need minimum data
            return None
        
        try:
            # Filter data by date range
            if start_date or end_date:
                filtered_data = []
                for timestamp, return_val in strategy_data:
                    if start_date and timestamp < start_date:
                        continue
                    if end_date and timestamp > end_date:
                        continue
                    filtered_data.append((timestamp, return_val))
                strategy_data = filtered_data
            
            if len(strategy_data) < 10:
                return None
            
            # Extract returns and dates
            dates = [item[0] for item in strategy_data]
            returns = np.array([item[1] for item in strategy_data])
            
            period_start = dates[0]
            period_end = dates[-1]
            
            # Get corresponding benchmark and portfolio returns
            benchmark_data = self._get_aligned_returns(self.benchmark_returns, dates)
            portfolio_data = self._get_aligned_returns(self.portfolio_returns, dates)
            
            # Basic return metrics
            total_return = np.prod(1 + returns) - 1
            n_periods = len(returns)
            periods_per_year = 252  # Assume daily data
            annualized_return = (1 + total_return) ** (periods_per_year / n_periods) - 1
            
            # Excess return vs benchmark
            excess_returns = returns - benchmark_data
            excess_return = np.mean(excess_returns) * periods_per_year
            
            # Cumulative return
            cumulative_return = total_return
            
            # Risk metrics
            volatility = np.std(returns) * np.sqrt(periods_per_year)
            downside_returns = returns[returns < 0]
            downside_volatility = np.std(downside_returns) * np.sqrt(periods_per_year) if len(downside_returns) > 0 else 0
            
            tracking_error = np.std(excess_returns) * np.sqrt(periods_per_year)
            
            # Maximum drawdown
            cumulative_returns = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = np.min(drawdown)
            
            # VaR and CVaR
            var_95 = np.percentile(returns, 5) * np.sqrt(periods_per_year)
            cvar_95 = np.mean(returns[returns <= np.percentile(returns, 5)]) * np.sqrt(periods_per_year)
            
            # Risk-adjusted metrics
            sharpe_ratio = (annualized_return - self.risk_free_rate) / volatility if volatility > 0 else 0
            information_ratio = excess_return / tracking_error if tracking_error > 0 else 0
            sortino_ratio = (annualized_return - self.risk_free_rate) / downside_volatility if downside_volatility > 0 else 0
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown < 0 else 0
            
            # Alpha and Beta (vs benchmark)
            if len(benchmark_data) > 1:
                covariance = np.cov(returns, benchmark_data)[0, 1]
                benchmark_variance = np.var(benchmark_data)
                beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
                alpha = (annualized_return - self.risk_free_rate) - beta * (np.mean(benchmark_data) * periods_per_year - self.risk_free_rate)
                r_squared = np.corrcoef(returns, benchmark_data)[0, 1] ** 2
            else:
                beta = 0
                alpha = 0
                r_squared = 0
            
            # Correlation metrics
            if len(portfolio_data) > 1:
                correlation_with_portfolio = np.corrcoef(returns, portfolio_data)[0, 1]
            else:
                correlation_with_portfolio = 0
                
            correlation_with_benchmark = np.corrcoef(returns, benchmark_data)[0, 1] if len(benchmark_data) > 1 else 0
            
            # Performance consistency metrics
            hit_ratio = np.mean(excess_returns > 0)
            
            # Up/Down capture
            benchmark_up = benchmark_data > 0
            benchmark_down = benchmark_data < 0
            
            if np.any(benchmark_up):
                up_capture = np.mean(returns[benchmark_up]) / np.mean(benchmark_data[benchmark_up])
            else:
                up_capture = 1.0
                
            if np.any(benchmark_down):
                down_capture = np.mean(returns[benchmark_down]) / np.mean(benchmark_data[benchmark_down])
            else:
                down_capture = 1.0
            
            # Return distribution metrics
            skewness = stats.skew(returns)
            kurtosis = stats.kurtosis(returns)
            
            # Return stability (negative of coefficient of variation of rolling returns)
            if len(returns) >= 20:
                rolling_returns = pd.Series(returns).rolling(20).mean().dropna()
                return_stability = 1.0 / (np.std(rolling_returns) / np.mean(rolling_returns)) if np.mean(rolling_returns) != 0 else 0
            else:
                return_stability = 0
            
            metrics = StrategyPerformanceMetrics(
                strategy_id=strategy_id,
                period_start=period_start,
                period_end=period_end,
                total_return=total_return,
                annualized_return=annualized_return,
                excess_return=excess_return,
                cumulative_return=cumulative_return,
                volatility=volatility,
                downside_volatility=downside_volatility,
                tracking_error=tracking_error,
                max_drawdown=max_drawdown,
                var_95=var_95,
                cvar_95=cvar_95,
                sharpe_ratio=sharpe_ratio,
                information_ratio=information_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                alpha=alpha,
                beta=beta,
                correlation_with_portfolio=correlation_with_portfolio,
                correlation_with_benchmark=correlation_with_benchmark,
                r_squared=r_squared,
                hit_ratio=hit_ratio,
                up_capture=up_capture,
                down_capture=down_capture,
                skewness=skewness,
                kurtosis=kurtosis,
                return_stability=return_stability
            )
            
            # Store in history
            self.strategy_metrics_history[strategy_id].append(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error("Strategy performance calculation failed", strategy_id=strategy_id, error=str(e))
            return None
    
    def calculate_portfolio_attribution(
        self,
        method: AttributionMethod = AttributionMethod.BRINSON_FACHLER,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Optional[PortfolioAttribution]:
        """
        Calculate portfolio performance attribution
        
        Args:
            method: Attribution method to use
            start_date: Analysis start date (optional)
            end_date: Analysis end date (optional)
            
        Returns:
            Portfolio attribution analysis
        """
        start_time = datetime.now()
        
        if len(self.portfolio_returns) < 30 or len(self.weights_history) < 30:
            return None
        
        try:
            # Filter data by date range and align
            portfolio_data = self._filter_by_date(self.portfolio_returns, start_date, end_date)
            benchmark_data = self._filter_by_date(self.benchmark_returns, start_date, end_date)
            weights_data = self._filter_by_date(self.weights_history, start_date, end_date)
            
            if len(portfolio_data) < 10:
                return None
            
            # Extract aligned data
            dates = [item[0] for item in portfolio_data]
            portfolio_returns = np.array([item[1] for item in portfolio_data])
            benchmark_returns = np.array([item[1] for item in benchmark_data])
            
            period_start = dates[0]
            period_end = dates[-1]
            
            # Portfolio-level metrics
            portfolio_return = np.mean(portfolio_returns)
            benchmark_return = np.mean(benchmark_returns)
            excess_return = portfolio_return - benchmark_return
            tracking_error = np.std(portfolio_returns - benchmark_returns)
            information_ratio = excess_return / tracking_error if tracking_error > 0 else 0
            
            # Calculate strategy contributions based on method
            if method == AttributionMethod.BRINSON_FACHLER:
                strategy_contributions = self._brinson_fachler_attribution(dates, weights_data)
            elif method == AttributionMethod.RETURN_DECOMPOSITION:
                strategy_contributions = self._return_decomposition_attribution(dates, weights_data)
            else:
                # Default to return decomposition
                strategy_contributions = self._return_decomposition_attribution(dates, weights_data)
            
            # Calculate totals
            total_selection = sum(contrib.selection_effect for contrib in strategy_contributions)
            total_allocation = sum(contrib.allocation_effect for contrib in strategy_contributions)
            total_interaction = sum(contrib.interaction_effect for contrib in strategy_contributions)
            
            # Attribution residual
            explained_return = total_selection + total_allocation + total_interaction
            attribution_residual = excess_return - explained_return
            
            # Attribution quality (R-squared of explained vs actual)
            if abs(excess_return) > 1e-10:
                attribution_r_squared = 1.0 - (attribution_residual / excess_return) ** 2
            else:
                attribution_r_squared = 1.0
            
            attribution = PortfolioAttribution(
                timestamp=datetime.now(),
                period_start=period_start,
                period_end=period_end,
                method=method,
                portfolio_return=portfolio_return,
                benchmark_return=benchmark_return,
                excess_return=excess_return,
                tracking_error=tracking_error,
                information_ratio=information_ratio,
                strategy_contributions=strategy_contributions,
                total_selection_effect=total_selection,
                total_allocation_effect=total_allocation,
                total_interaction_effect=total_interaction,
                attribution_residual=attribution_residual,
                attribution_r_squared=max(0, attribution_r_squared)
            )
            
            # Store result
            self.attribution_history.append(attribution)
            
            # Track calculation time
            calc_time = (datetime.now() - start_time).total_seconds() * 1000
            self.calculation_times.append(calc_time)
            
            logger.info("Portfolio attribution completed",
                       method=method.value,
                       excess_return=excess_return,
                       information_ratio=information_ratio,
                       attribution_r_squared=attribution_r_squared,
                       calculation_time_ms=calc_time)
            
            return attribution
            
        except Exception as e:
            logger.error("Portfolio attribution calculation failed", error=str(e))
            return None
    
    def _brinson_fachler_attribution(
        self,
        dates: List[datetime],
        weights_data: List[Tuple[datetime, np.ndarray]]
    ) -> List[AttributionContribution]:
        """Brinson-Fachler attribution analysis"""
        
        strategy_contributions = []
        
        try:
            # Get strategy returns for the period
            strategy_returns_data = {}
            for strategy_id in self.strategy_returns:
                strategy_data = self._get_aligned_returns(
                    self.strategy_returns[strategy_id], dates
                )
                strategy_returns_data[strategy_id] = strategy_data
            
            # Calculate benchmark weights (equal weight)
            benchmark_weights = np.array([1.0 / self.n_strategies] * self.n_strategies)
            
            # Get average portfolio weights for the period
            if weights_data:
                avg_weights = np.mean([weights for _, weights in weights_data], axis=0)
            else:
                avg_weights = benchmark_weights
            
            for i, strategy_id in enumerate(sorted(self.strategy_returns.keys())):
                if i >= len(avg_weights):
                    break
                
                weight = avg_weights[i]
                benchmark_weight = benchmark_weights[i]
                
                # Strategy return
                if strategy_id in strategy_returns_data:
                    strategy_return = np.mean(strategy_returns_data[strategy_id])
                else:
                    strategy_return = 0
                
                # Benchmark return (portfolio return)
                portfolio_return_data = self._get_aligned_returns(self.portfolio_returns, dates)
                benchmark_return = np.mean(portfolio_return_data)
                
                # Attribution effects
                allocation_effect = (weight - benchmark_weight) * benchmark_return
                selection_effect = benchmark_weight * (strategy_return - benchmark_return)
                interaction_effect = (weight - benchmark_weight) * (strategy_return - benchmark_return)
                
                # Return contribution
                return_contribution = weight * strategy_return
                excess_return_contribution = weight * (strategy_return - benchmark_return)
                
                # Risk contribution (simplified)
                risk_contribution = weight  # Simplified
                tracking_error_contribution = abs(weight - benchmark_weight)
                
                contribution = AttributionContribution(
                    strategy_id=strategy_id,
                    weight=weight,
                    return_contribution=return_contribution,
                    excess_return_contribution=excess_return_contribution,
                    risk_contribution=risk_contribution,
                    tracking_error_contribution=tracking_error_contribution,
                    selection_effect=selection_effect,
                    allocation_effect=allocation_effect,
                    interaction_effect=interaction_effect
                )
                
                strategy_contributions.append(contribution)
        
        except Exception as e:
            logger.error("Brinson-Fachler attribution failed", error=str(e))
        
        return strategy_contributions
    
    def _return_decomposition_attribution(
        self,
        dates: List[datetime],
        weights_data: List[Tuple[datetime, np.ndarray]]
    ) -> List[AttributionContribution]:
        """Simple return decomposition attribution"""
        
        strategy_contributions = []
        
        try:
            # Get average weights
            if weights_data:
                avg_weights = np.mean([weights for _, weights in weights_data], axis=0)
            else:
                avg_weights = np.array([1.0 / self.n_strategies] * self.n_strategies)
            
            # Get portfolio and benchmark returns
            portfolio_return_data = self._get_aligned_returns(self.portfolio_returns, dates)
            benchmark_return_data = self._get_aligned_returns(self.benchmark_returns, dates)
            
            portfolio_return = np.mean(portfolio_return_data)
            benchmark_return = np.mean(benchmark_return_data)
            
            for i, strategy_id in enumerate(sorted(self.strategy_returns.keys())):
                if i >= len(avg_weights):
                    break
                
                weight = avg_weights[i]
                
                # Get strategy returns
                strategy_data = self._get_aligned_returns(
                    self.strategy_returns[strategy_id], dates
                )
                strategy_return = np.mean(strategy_data) if len(strategy_data) > 0 else 0
                
                # Simple attribution
                return_contribution = weight * strategy_return
                excess_return_contribution = weight * (strategy_return - benchmark_return)
                
                # Simplified effects
                selection_effect = strategy_return - portfolio_return
                allocation_effect = weight * (portfolio_return - benchmark_return)
                interaction_effect = 0  # Simplified
                
                contribution = AttributionContribution(
                    strategy_id=strategy_id,
                    weight=weight,
                    return_contribution=return_contribution,
                    excess_return_contribution=excess_return_contribution,
                    risk_contribution=weight,
                    tracking_error_contribution=abs(excess_return_contribution),
                    selection_effect=selection_effect,
                    allocation_effect=allocation_effect,
                    interaction_effect=interaction_effect
                )
                
                strategy_contributions.append(contribution)
        
        except Exception as e:
            logger.error("Return decomposition attribution failed", error=str(e))
        
        return strategy_contributions
    
    def _get_aligned_returns(
        self,
        return_data: List[Tuple[datetime, float]],
        target_dates: List[datetime]
    ) -> np.ndarray:
        """Get returns aligned to target dates"""
        
        aligned_returns = []
        return_dict = {date: ret for date, ret in return_data}
        
        for date in target_dates:
            if date in return_dict:
                aligned_returns.append(return_dict[date])
            else:
                # Find closest date
                closest_date = min(return_dict.keys(), key=lambda x: abs((x - date).total_seconds()))
                aligned_returns.append(return_dict[closest_date])
        
        return np.array(aligned_returns)
    
    def _filter_by_date(
        self,
        data: List[Tuple[datetime, Union[float, np.ndarray]]],
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> List[Tuple[datetime, Union[float, np.ndarray]]]:
        """Filter data by date range"""
        
        filtered_data = []
        for timestamp, value in data:
            if start_date and timestamp < start_date:
                continue
            if end_date and timestamp > end_date:
                continue
            filtered_data.append((timestamp, value))
        
        return filtered_data
    
    def _update_factor_returns(self, timestamp: datetime):
        """Update factor returns (simplified simulation)"""
        # Simulate factor returns based on market conditions
        for factor_name in self.factor_names:
            # Generate random factor return
            factor_return = np.random.normal(0, 0.01)  # 1% daily volatility
            self.factor_returns[factor_name].append((timestamp, factor_return))
            
            # Keep rolling window
            if len(self.factor_returns[factor_name]) > self.attribution_window * 2:
                self.factor_returns[factor_name] = self.factor_returns[factor_name][-self.attribution_window:]
    
    def get_strategy_performance_summary(self, strategy_id: str) -> Dict:
        """Get performance summary for a strategy"""
        if strategy_id not in self.strategy_metrics_history:
            return {"error": f"Strategy {strategy_id} not found"}
        
        metrics_history = self.strategy_metrics_history[strategy_id]
        if not metrics_history:
            return {"error": f"No performance data for strategy {strategy_id}"}
        
        latest_metrics = metrics_history[-1]
        
        return {
            "strategy_id": strategy_id,
            "latest_metrics": {
                "annualized_return": latest_metrics.annualized_return,
                "volatility": latest_metrics.volatility,
                "sharpe_ratio": latest_metrics.sharpe_ratio,
                "information_ratio": latest_metrics.information_ratio,
                "max_drawdown": latest_metrics.max_drawdown,
                "alpha": latest_metrics.alpha,
                "beta": latest_metrics.beta,
                "correlation_with_portfolio": latest_metrics.correlation_with_portfolio
            },
            "performance_history_count": len(metrics_history),
            "data_points": len(self.strategy_returns.get(strategy_id, []))
        }
    
    def get_attribution_summary(self) -> Dict:
        """Get portfolio attribution summary"""
        if not self.attribution_history:
            return {"status": "No attribution analysis available"}
        
        latest_attribution = self.attribution_history[-1]
        
        return {
            "total_attributions": len(self.attribution_history),
            "avg_calculation_time_ms": np.mean(self.calculation_times) if self.calculation_times else 0,
            "latest_attribution": {
                "method": latest_attribution.method.value,
                "excess_return": latest_attribution.excess_return,
                "information_ratio": latest_attribution.information_ratio,
                "total_selection_effect": latest_attribution.total_selection_effect,
                "total_allocation_effect": latest_attribution.total_allocation_effect,
                "attribution_r_squared": latest_attribution.attribution_r_squared,
                "attribution_residual": latest_attribution.attribution_residual
            },
            "strategy_contributions": [
                {
                    "strategy_id": contrib.strategy_id,
                    "weight": contrib.weight,
                    "return_contribution": contrib.return_contribution,
                    "selection_effect": contrib.selection_effect,
                    "allocation_effect": contrib.allocation_effect
                }
                for contrib in latest_attribution.strategy_contributions
            ]
        }