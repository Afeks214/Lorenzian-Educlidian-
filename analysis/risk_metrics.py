"""
Risk Metrics Module with VaR Calculator Integration

This module provides advanced risk metrics that integrate with the existing
VaR calculation system, including:
- Expected Shortfall (ES) calculations
- Regime-aware risk adjustments
- Component VaR and Marginal VaR
- Integration with correlation tracker
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
import logging
from numba import jit, njit
from scipy import stats
import warnings

# Import existing VaR system components
try:
    from src.risk.core.var_calculator import VaRCalculator, VaRResult, PositionData
    from src.risk.core.correlation_tracker import CorrelationTracker, CorrelationRegime
    from src.core.events import EventBus
    VAR_SYSTEM_AVAILABLE = True
except ImportError:
    VAR_SYSTEM_AVAILABLE = False
    VaRCalculator = None
    VaRResult = None
    PositionData = None
    CorrelationTracker = None
    EventBus = None

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    """Container for comprehensive risk metrics"""
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    expected_shortfall: float
    maximum_drawdown: float
    volatility: float
    downside_volatility: float
    beta: float
    tracking_error: float
    information_ratio: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    regime_adjusted_var: float
    component_var: Dict[str, float]
    marginal_var: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'var_95': self.var_95,
            'var_99': self.var_99,
            'cvar_95': self.cvar_95,
            'cvar_99': self.cvar_99,
            'expected_shortfall': self.expected_shortfall,
            'maximum_drawdown': self.maximum_drawdown,
            'volatility': self.volatility,
            'downside_volatility': self.downside_volatility,
            'beta': self.beta,
            'tracking_error': self.tracking_error,
            'information_ratio': self.information_ratio,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'regime_adjusted_var': self.regime_adjusted_var,
            'component_var': self.component_var,
            'marginal_var': self.marginal_var
        }


@dataclass
class RegimeRiskMetrics:
    """Risk metrics by correlation regime"""
    normal_regime: RiskMetrics
    elevated_regime: RiskMetrics
    crisis_regime: RiskMetrics
    shock_regime: RiskMetrics
    current_regime: str
    
    def get_current_metrics(self) -> RiskMetrics:
        """Get metrics for current regime"""
        regime_map = {
            'NORMAL': self.normal_regime,
            'ELEVATED': self.elevated_regime,
            'CRISIS': self.crisis_regime,
            'SHOCK': self.shock_regime
        }
        return regime_map.get(self.current_regime, self.normal_regime)


@dataclass
class ComponentRiskAnalysis:
    """Component-level risk analysis"""
    asset_symbol: str
    component_var: float
    marginal_var: float
    risk_contribution: float
    correlation_impact: float
    volatility_impact: float
    concentration_risk: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            'asset_symbol': self.asset_symbol,
            'component_var': self.component_var,
            'marginal_var': self.marginal_var,
            'risk_contribution': self.risk_contribution,
            'correlation_impact': self.correlation_impact,
            'volatility_impact': self.volatility_impact,
            'concentration_risk': self.concentration_risk
        }


class RiskMetricsCalculator:
    """
    Advanced risk metrics calculator with VaR system integration.
    
    Provides comprehensive risk analysis including VaR, CVaR, component
    risk metrics, and regime-aware risk adjustments.
    """
    
    def __init__(
        self,
        var_calculator: Optional[VaRCalculator] = None,
        correlation_tracker: Optional[CorrelationTracker] = None,
        event_bus: Optional[EventBus] = None,
        confidence_levels: List[float] = [0.95, 0.99],
        time_horizons: List[int] = [1, 10],
        risk_free_rate: float = 0.0,
        target_volatility: float = 0.15
    ):
        """
        Initialize risk metrics calculator
        
        Args:
            var_calculator: VaR calculator instance
            correlation_tracker: Correlation tracker instance
            event_bus: Event bus for notifications
            confidence_levels: Confidence levels for VaR/CVaR
            time_horizons: Time horizons in days
            risk_free_rate: Annual risk-free rate
            target_volatility: Target volatility for scaling
        """
        self.var_calculator = var_calculator
        self.correlation_tracker = correlation_tracker
        self.event_bus = event_bus
        self.confidence_levels = confidence_levels
        self.time_horizons = time_horizons
        self.risk_free_rate = risk_free_rate
        self.target_volatility = target_volatility
        
        # Risk metrics history
        self.risk_history: List[RiskMetrics] = []
        
        # Performance monitoring
        self.calculation_times: List[float] = []
        self.performance_target_ms = 100.0  # Target 100ms for risk metrics
        
        # Regime adjustments
        self.regime_multipliers = {
            'NORMAL': 1.0,
            'ELEVATED': 1.2,
            'CRISIS': 1.5,
            'SHOCK': 2.0
        }
        
        logger.info("RiskMetricsCalculator initialized",
                   extra={
                       'var_calculator_available': var_calculator is not None,
                       'correlation_tracker_available': correlation_tracker is not None,
                       'confidence_levels': confidence_levels,
                       'time_horizons': time_horizons
                   })
    
    @njit
    def _calculate_expected_shortfall(
        self,
        returns: np.ndarray,
        confidence_level: float
    ) -> float:
        """Calculate Expected Shortfall (CVaR) - JIT optimized"""
        if len(returns) == 0:
            return 0.0
        
        # Sort returns
        sorted_returns = np.sort(returns)
        
        # Find VaR threshold
        var_index = int((1 - confidence_level) * len(sorted_returns))
        if var_index >= len(sorted_returns):
            var_index = len(sorted_returns) - 1
        
        # Expected Shortfall is mean of tail beyond VaR
        tail_returns = sorted_returns[:var_index + 1]
        
        if len(tail_returns) == 0:
            return 0.0
        
        return abs(np.mean(tail_returns))
    
    @njit
    def _calculate_maximum_drawdown(self, equity_curve: np.ndarray) -> float:
        """Calculate maximum drawdown - JIT optimized"""
        if len(equity_curve) == 0:
            return 0.0
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(equity_curve)
        
        # Calculate drawdown
        drawdown = (equity_curve - running_max) / running_max
        
        # Return maximum drawdown
        return abs(np.min(drawdown))
    
    @njit
    def _calculate_downside_volatility(
        self,
        returns: np.ndarray,
        target_return: float = 0.0
    ) -> float:
        """Calculate downside volatility - JIT optimized"""
        if len(returns) == 0:
            return 0.0
        
        # Get downside returns
        downside_returns = returns[returns < target_return]
        
        if len(downside_returns) == 0:
            return 0.0
        
        # Calculate downside deviation
        downside_variance = np.mean((downside_returns - target_return) ** 2)
        
        return np.sqrt(downside_variance)
    
    @njit
    def _calculate_tracking_error(
        self,
        returns: np.ndarray,
        benchmark_returns: np.ndarray
    ) -> float:
        """Calculate tracking error - JIT optimized"""
        if len(returns) == 0 or len(benchmark_returns) == 0:
            return 0.0
        
        # Ensure same length
        min_len = min(len(returns), len(benchmark_returns))
        returns = returns[:min_len]
        benchmark_returns = benchmark_returns[:min_len]
        
        # Calculate excess returns
        excess_returns = returns - benchmark_returns
        
        # Tracking error is standard deviation of excess returns
        return np.std(excess_returns)
    
    @njit
    def _calculate_beta(
        self,
        returns: np.ndarray,
        benchmark_returns: np.ndarray
    ) -> float:
        """Calculate beta - JIT optimized"""
        if len(returns) == 0 or len(benchmark_returns) == 0:
            return 1.0
        
        # Ensure same length
        min_len = min(len(returns), len(benchmark_returns))
        returns = returns[:min_len]
        benchmark_returns = benchmark_returns[:min_len]
        
        # Calculate covariance and variance
        covariance = np.mean((returns - np.mean(returns)) * (benchmark_returns - np.mean(benchmark_returns)))
        benchmark_variance = np.var(benchmark_returns)
        
        if benchmark_variance == 0:
            return 1.0
        
        return covariance / benchmark_variance
    
    async def calculate_comprehensive_risk_metrics(
        self,
        returns: np.ndarray,
        equity_curve: np.ndarray,
        benchmark_returns: Optional[np.ndarray] = None,
        asset_positions: Optional[Dict[str, PositionData]] = None,
        periods_per_year: int = 252
    ) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics
        
        Args:
            returns: Array of portfolio returns
            equity_curve: Array of portfolio values
            benchmark_returns: Optional benchmark returns
            asset_positions: Optional position data for component analysis
            periods_per_year: Number of periods per year
        
        Returns:
            RiskMetrics object
        """
        start_time = datetime.now()
        
        try:
            # Basic risk metrics
            volatility = np.std(returns) * np.sqrt(periods_per_year) if len(returns) > 0 else 0.0
            downside_volatility = self._calculate_downside_volatility(returns) * np.sqrt(periods_per_year)
            maximum_drawdown = self._calculate_maximum_drawdown(equity_curve)
            
            # VaR and CVaR calculations
            var_95 = 0.0
            var_99 = 0.0
            cvar_95 = 0.0
            cvar_99 = 0.0
            regime_adjusted_var = 0.0
            
            if self.var_calculator is not None and VAR_SYSTEM_AVAILABLE:
                # Use integrated VaR calculator
                try:
                    var_result_95 = await self.var_calculator.calculate_var(
                        confidence_level=0.95,
                        time_horizon=1,
                        method="parametric"
                    )
                    
                    var_result_99 = await self.var_calculator.calculate_var(
                        confidence_level=0.99,
                        time_horizon=1,
                        method="parametric"
                    )
                    
                    if var_result_95:
                        var_95 = var_result_95.portfolio_var
                        regime_adjusted_var = var_95
                    
                    if var_result_99:
                        var_99 = var_result_99.portfolio_var
                        
                except Exception as e:
                    logger.warning(f"VaR calculation failed: {e}")
            
            # Calculate CVaR using historical method
            if len(returns) > 0:
                cvar_95 = self._calculate_expected_shortfall(returns, 0.95)
                cvar_99 = self._calculate_expected_shortfall(returns, 0.99)
                
                # Fallback VaR calculation if external calculator failed
                if var_95 == 0.0:
                    sorted_returns = np.sort(returns)
                    var_95_idx = int(0.05 * len(sorted_returns))
                    var_99_idx = int(0.01 * len(sorted_returns))
                    
                    if var_95_idx < len(sorted_returns):
                        var_95 = abs(sorted_returns[var_95_idx])
                    if var_99_idx < len(sorted_returns):
                        var_99 = abs(sorted_returns[var_99_idx])
            
            # Expected Shortfall (same as CVaR_95)
            expected_shortfall = cvar_95
            
            # Benchmark-related metrics
            beta = 1.0
            tracking_error = 0.0
            information_ratio = 0.0
            
            if benchmark_returns is not None and len(benchmark_returns) > 0:
                beta = self._calculate_beta(returns, benchmark_returns)
                tracking_error = self._calculate_tracking_error(returns, benchmark_returns)
                
                # Information ratio
                if tracking_error > 0:
                    excess_returns = returns - benchmark_returns
                    information_ratio = np.mean(excess_returns) / tracking_error * np.sqrt(periods_per_year)
            
            # Performance ratios
            excess_returns = returns - self.risk_free_rate / periods_per_year
            
            # Sharpe ratio
            sharpe_ratio = 0.0
            if volatility > 0:
                sharpe_ratio = np.mean(excess_returns) / volatility * np.sqrt(periods_per_year)
            
            # Sortino ratio
            sortino_ratio = 0.0
            if downside_volatility > 0:
                sortino_ratio = np.mean(excess_returns) / downside_volatility * np.sqrt(periods_per_year)
            
            # Calmar ratio
            calmar_ratio = 0.0
            if maximum_drawdown > 0:
                annual_return = np.mean(returns) * periods_per_year
                calmar_ratio = annual_return / maximum_drawdown
            
            # Component and marginal VaR
            component_var = {}
            marginal_var = {}
            
            if (self.var_calculator is not None and 
                VAR_SYSTEM_AVAILABLE and 
                hasattr(self.var_calculator, 'get_latest_var')):
                
                try:
                    latest_var = self.var_calculator.get_latest_var()
                    if latest_var:
                        component_var = latest_var.component_vars
                        marginal_var = latest_var.marginal_vars
                except Exception as e:
                    logger.warning(f"Component VaR extraction failed: {e}")
            
            # Regime adjustment
            if self.correlation_tracker is not None and hasattr(self.correlation_tracker, 'current_regime'):
                current_regime = self.correlation_tracker.current_regime
                if hasattr(current_regime, 'value'):
                    regime_name = current_regime.value
                    multiplier = self.regime_multipliers.get(regime_name, 1.0)
                    regime_adjusted_var = var_95 * multiplier
            
            # Create risk metrics object
            risk_metrics = RiskMetrics(
                var_95=var_95,
                var_99=var_99,
                cvar_95=cvar_95,
                cvar_99=cvar_99,
                expected_shortfall=expected_shortfall,
                maximum_drawdown=maximum_drawdown,
                volatility=volatility,
                downside_volatility=downside_volatility,
                beta=beta,
                tracking_error=tracking_error,
                information_ratio=information_ratio,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                regime_adjusted_var=regime_adjusted_var,
                component_var=component_var,
                marginal_var=marginal_var
            )
            
            # Store in history
            self.risk_history.append(risk_metrics)
            
            # Keep only recent history
            if len(self.risk_history) > 1000:
                self.risk_history = self.risk_history[-1000:]
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Risk metrics calculation failed: {e}")
            # Return empty metrics
            return RiskMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, {}, {})
        
        finally:
            # Track performance
            calc_time = (datetime.now() - start_time).total_seconds() * 1000
            self.calculation_times.append(calc_time)
            
            if len(self.calculation_times) > 100:
                self.calculation_times = self.calculation_times[-100:]
    
    def analyze_component_risk(
        self,
        asset_positions: Dict[str, PositionData],
        latest_var_result: Optional[VaRResult] = None
    ) -> Dict[str, ComponentRiskAnalysis]:
        """
        Analyze component-level risk contributions
        
        Args:
            asset_positions: Dictionary of asset positions
            latest_var_result: Latest VaR calculation result
        
        Returns:
            Dictionary of asset symbol -> ComponentRiskAnalysis
        """
        component_analysis = {}
        
        if not asset_positions:
            return component_analysis
        
        # Calculate portfolio value
        portfolio_value = sum(pos.market_value for pos in asset_positions.values())
        
        if portfolio_value == 0:
            return component_analysis
        
        # Get component VaR from latest result
        component_vars = {}
        marginal_vars = {}
        
        if latest_var_result:
            component_vars = latest_var_result.component_vars
            marginal_vars = latest_var_result.marginal_vars
        
        # Analyze each asset
        for symbol, position in asset_positions.items():
            weight = position.market_value / portfolio_value
            
            # Component VaR
            component_var = component_vars.get(symbol, 0.0)
            marginal_var = marginal_vars.get(symbol, 0.0)
            
            # Risk contribution as percentage of total VaR
            risk_contribution = 0.0
            if latest_var_result and latest_var_result.portfolio_var > 0:
                risk_contribution = component_var / latest_var_result.portfolio_var
            
            # Volatility impact
            volatility_impact = position.volatility * weight
            
            # Correlation impact (simplified)
            correlation_impact = 0.0
            if self.correlation_tracker and hasattr(self.correlation_tracker, 'get_correlation_matrix'):
                try:
                    correlation_matrix = self.correlation_tracker.get_correlation_matrix()
                    if correlation_matrix is not None:
                        # Simplified correlation impact calculation
                        correlation_impact = np.mean(correlation_matrix) * weight
                except Exception as e:
                    logger.warning(f"Correlation impact calculation failed: {e}")
            
            # Concentration risk
            concentration_risk = weight ** 2  # Simplified Herfindahl index contribution
            
            # Create component analysis
            component_analysis[symbol] = ComponentRiskAnalysis(
                asset_symbol=symbol,
                component_var=component_var,
                marginal_var=marginal_var,
                risk_contribution=risk_contribution,
                correlation_impact=correlation_impact,
                volatility_impact=volatility_impact,
                concentration_risk=concentration_risk
            )
        
        return component_analysis
    
    def calculate_regime_aware_metrics(
        self,
        returns: np.ndarray,
        equity_curve: np.ndarray,
        benchmark_returns: Optional[np.ndarray] = None
    ) -> RegimeRiskMetrics:
        """
        Calculate risk metrics by correlation regime
        
        Args:
            returns: Array of portfolio returns
            equity_curve: Array of portfolio values
            benchmark_returns: Optional benchmark returns
        
        Returns:
            RegimeRiskMetrics object
        """
        # Base metrics calculation
        base_metrics = asyncio.run(self.calculate_comprehensive_risk_metrics(
            returns, equity_curve, benchmark_returns
        ))
        
        # Create regime-adjusted metrics
        regime_metrics = {}
        
        for regime_name, multiplier in self.regime_multipliers.items():
            # Apply regime multiplier to risk metrics
            adjusted_metrics = RiskMetrics(
                var_95=base_metrics.var_95 * multiplier,
                var_99=base_metrics.var_99 * multiplier,
                cvar_95=base_metrics.cvar_95 * multiplier,
                cvar_99=base_metrics.cvar_99 * multiplier,
                expected_shortfall=base_metrics.expected_shortfall * multiplier,
                maximum_drawdown=base_metrics.maximum_drawdown,  # Drawdown doesn't scale
                volatility=base_metrics.volatility,  # Volatility doesn't scale
                downside_volatility=base_metrics.downside_volatility,
                beta=base_metrics.beta,
                tracking_error=base_metrics.tracking_error,
                information_ratio=base_metrics.information_ratio,
                sharpe_ratio=base_metrics.sharpe_ratio,  # Ratios don't scale
                sortino_ratio=base_metrics.sortino_ratio,
                calmar_ratio=base_metrics.calmar_ratio,
                regime_adjusted_var=base_metrics.var_95 * multiplier,
                component_var={k: v * multiplier for k, v in base_metrics.component_var.items()},
                marginal_var={k: v * multiplier for k, v in base_metrics.marginal_var.items()}
            )
            regime_metrics[regime_name] = adjusted_metrics
        
        # Get current regime
        current_regime = 'NORMAL'
        if self.correlation_tracker and hasattr(self.correlation_tracker, 'current_regime'):
            try:
                current_regime = self.correlation_tracker.current_regime.value
            except (ValueError, TypeError, AttributeError, KeyError) as e:
                logger.error(f'Error occurred: {e}')
        
        return RegimeRiskMetrics(
            normal_regime=regime_metrics['NORMAL'],
            elevated_regime=regime_metrics['ELEVATED'],
            crisis_regime=regime_metrics['CRISIS'],
            shock_regime=regime_metrics['SHOCK'],
            current_regime=current_regime
        )
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary"""
        if not self.risk_history:
            return {"status": "No risk metrics calculated"}
        
        latest_metrics = self.risk_history[-1]
        
        # Calculate performance statistics
        avg_calc_time = np.mean(self.calculation_times) if self.calculation_times else 0
        target_met = avg_calc_time <= self.performance_target_ms
        
        return {
            "timestamp": datetime.now().isoformat(),
            "latest_metrics": latest_metrics.to_dict(),
            "performance": {
                "avg_calc_time_ms": avg_calc_time,
                "target_met": target_met,
                "calculation_count": len(self.calculation_times),
                "performance_target_ms": self.performance_target_ms
            },
            "system_integration": {
                "var_calculator_available": self.var_calculator is not None,
                "correlation_tracker_available": self.correlation_tracker is not None,
                "var_system_available": VAR_SYSTEM_AVAILABLE
            },
            "risk_metrics_history_count": len(self.risk_history)
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self.calculation_times:
            return {
                "avg_calc_time_ms": 0,
                "max_calc_time_ms": 0,
                "target_met": True,
                "calculation_count": 0
            }
        
        avg_time = np.mean(self.calculation_times)
        max_time = np.max(self.calculation_times)
        target_met = avg_time <= self.performance_target_ms
        
        return {
            "avg_calc_time_ms": avg_time,
            "max_calc_time_ms": max_time,
            "target_met": target_met,
            "calculation_count": len(self.calculation_times),
            "performance_target_ms": self.performance_target_ms
        }


# Convenience functions for standalone usage
def calculate_var_historical(returns: np.ndarray, confidence_level: float = 0.95) -> float:
    """Calculate historical VaR"""
    if len(returns) == 0:
        return 0.0
    
    sorted_returns = np.sort(returns)
    index = int((1 - confidence_level) * len(sorted_returns))
    
    if index >= len(sorted_returns):
        index = len(sorted_returns) - 1
    
    return abs(sorted_returns[index])


def calculate_cvar_historical(returns: np.ndarray, confidence_level: float = 0.95) -> float:
    """Calculate historical CVaR"""
    if len(returns) == 0:
        return 0.0
    
    sorted_returns = np.sort(returns)
    var_index = int((1 - confidence_level) * len(sorted_returns))
    
    if var_index >= len(sorted_returns):
        var_index = len(sorted_returns) - 1
    
    tail_returns = sorted_returns[:var_index + 1]
    
    if len(tail_returns) == 0:
        return 0.0
    
    return abs(np.mean(tail_returns))


def calculate_expected_shortfall(returns: np.ndarray, confidence_level: float = 0.95) -> float:
    """Calculate Expected Shortfall (alias for CVaR)"""
    return calculate_cvar_historical(returns, confidence_level)