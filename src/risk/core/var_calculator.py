"""
Enhanced VaR Calculator with Adaptive Correlation Integration

This module implements a high-performance VaR calculator that integrates with
the enhanced correlation tracker for regime-aware risk measurement.

Key Features:
- Multiple VaR methodologies (Historical, Parametric, Monte Carlo)
- Integration with EWMA correlation tracker
- Real-time performance monitoring
- Regime-aware risk adjustments
- Portfolio-level and position-level VaR
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from scipy import stats
import asyncio
import structlog

from src.core.events import Event, EventType, EventBus
from src.risk.core.correlation_tracker import CorrelationTracker, CorrelationRegime
from src.algorithms.copula_models import CopulaVaRCalculator, create_copula_var_calculator, MarketRegime
from src.safety.trading_system_controller import get_controller

logger = structlog.get_logger()


@dataclass
class VaRResult:
    """VaR calculation result"""
    timestamp: datetime
    confidence_level: float
    time_horizon_days: int
    portfolio_var: float
    component_vars: Dict[str, float]
    marginal_vars: Dict[str, float]
    correlation_regime: str
    calculation_method: str
    performance_ms: float


@dataclass
class PositionData:
    """Position data for VaR calculation"""
    symbol: str
    quantity: float
    market_value: float
    price: float
    volatility: float  # Annualized volatility


class VaRCalculator:
    """
    Enhanced VaR calculator with adaptive correlation integration.
    
    Supports multiple VaR methodologies and integrates with correlation
    tracker for regime-aware risk measurement.
    """
    
    def __init__(
        self,
        correlation_tracker: CorrelationTracker,
        event_bus: EventBus,
        confidence_levels: List[float] = [0.95, 0.99],
        time_horizons: List[int] = [1, 10],  # Days
        default_method: str = "parametric"
    ):
        self.correlation_tracker = correlation_tracker
        self.event_bus = event_bus
        self.confidence_levels = confidence_levels
        self.time_horizons = time_horizons
        self.default_method = default_method
        
        # Enhanced risk metrics configuration
        self.calculate_expected_shortfall = True
        self.calculate_risk_attribution = True
        self.tail_risk_quantiles = [0.975, 0.995, 0.999]  # For tail risk analysis
        
        # Current portfolio state
        self.positions: Dict[str, PositionData] = {}
        self.portfolio_value: float = 0.0
        
        # VaR calculation history
        self.var_history: List[VaRResult] = []
        
        # Performance tracking
        self.performance_target_ms = 5.0
        self.calculation_times: List[float] = []
        
        # Regime adjustments
        self.regime_multipliers = {
            CorrelationRegime.NORMAL: 1.0,
            CorrelationRegime.ELEVATED: 1.2,
            CorrelationRegime.CRISIS: 1.5,
            CorrelationRegime.SHOCK: 2.0
        }
        
        # Initialize dynamic copula modeling
        self.copula_calculator = create_copula_var_calculator(
            confidence_levels=confidence_levels
        )
        self.enable_copula_modeling = True
        self.copula_cache = {}
        
        # Subscribe to events
        self._setup_event_subscriptions()
        
        # Register with trading system controller
        system_controller = get_controller()
        if system_controller:
            system_controller.register_component("var_calculator", {
                "confidence_levels": confidence_levels,
                "time_horizons": time_horizons,
                "default_method": default_method,
                "performance_target_ms": self.performance_target_ms
            })
        
        logger.info("VaRCalculator initialized",
                   confidence_levels=confidence_levels,
                   time_horizons=time_horizons,
                   default_method=default_method)
    
    def _setup_event_subscriptions(self):
        """Setup event subscriptions"""
        self.event_bus.subscribe(EventType.POSITION_UPDATE, self._handle_position_update)
        self.event_bus.subscribe(EventType.VAR_UPDATE, self._handle_correlation_update)
    
    def _handle_position_update(self, event: Event):
        """Handle position updates"""
        position_data = event.payload
        
        if hasattr(position_data, 'positions'):
            # Full portfolio update
            self.positions = {}
            total_value = 0.0
            
            for pos in position_data.positions:
                position = PositionData(
                    symbol=pos.symbol,
                    quantity=pos.quantity,
                    market_value=pos.market_value,
                    price=pos.price,
                    volatility=pos.volatility if hasattr(pos, 'volatility') else 0.2  # Default 20%
                )
                self.positions[pos.symbol] = position
                total_value += pos.market_value
            
            self.portfolio_value = total_value
            
            # Trigger VaR calculation
            asyncio.create_task(self._calculate_portfolio_var())
    
    def _handle_correlation_update(self, event: Event):
        """Handle correlation matrix updates"""
        # Trigger VaR recalculation when correlations change
        if self.positions:
            asyncio.create_task(self._calculate_portfolio_var())
    
    async def _calculate_portfolio_var(self):
        """Calculate portfolio VaR asynchronously"""
        # Check if system is ON before performing calculations
        system_controller = get_controller()
        if system_controller and not system_controller.is_system_on():
            logger.debug("System is OFF - skipping automatic portfolio VaR calculation")
            return
        
        start_time = datetime.now()
        
        try:
            # Calculate VaR for primary confidence level and horizon
            var_result = await self.calculate_var(
                confidence_level=self.confidence_levels[0],
                time_horizon=self.time_horizons[0],
                method=self.default_method
            )
            
            if var_result:
                # Store result
                self.var_history.append(var_result)
                
                # Keep only recent history for memory efficiency
                if len(self.var_history) > 1000:
                    self.var_history = self.var_history[-1000:]
                
                # Publish VaR update
                self.event_bus.publish(
                    self.event_bus.create_event(
                        EventType.VAR_UPDATE,
                        var_result,
                        'VaRCalculator'
                    )
                )
                
                # Check for VaR breaches
                self._check_var_breach(var_result)
        
        except Exception as e:
            logger.error("Error calculating portfolio VaR", error=str(e), exc_info=True)
            
            # Create system error for failed VaR calculation
            from src.core.errors.base_exceptions import SystemError, ErrorContext
            from src.core.errors.error_handler import get_error_handler
            
            error_handler = get_error_handler()
            context = ErrorContext(
                additional_data={
                    "portfolio_value": self.portfolio_value,
                    "position_count": len(self.positions),
                    "confidence_level": self.confidence_levels[0],
                    "calculation_method": self.default_method
                }
            )
            
            # Handle VaR calculation failure
            var_error = SystemError(
                message=f"VaR calculation failed: {str(e)}",
                context=context,
                cause=e,
                recoverable=True
            )
            
            error_handler.handle_exception(var_error, context, function_name="calculate_portfolio_var")
        
        finally:
            calc_time = (datetime.now() - start_time).total_seconds() * 1000
            self.calculation_times.append(calc_time)
            
            # Keep only recent performance data
            if len(self.calculation_times) > 100:
                self.calculation_times = self.calculation_times[-100:]
    
    async def calculate_var(
        self,
        confidence_level: float = 0.95,
        time_horizon: int = 1,
        method: str = "parametric",
        calculate_expected_shortfall: bool = True
    ) -> Optional[VaRResult]:
        """
        Calculate Value at Risk and Expected Shortfall for current portfolio.
        
        Args:
            confidence_level: Confidence level (0.95 = 95%)
            time_horizon: Time horizon in days
            method: Calculation method ('parametric', 'historical', 'monte_carlo')
            calculate_expected_shortfall: Whether to calculate ES metrics
        
        Returns:
            VaRResult with enhanced risk metrics or None if calculation fails
        """
        # Check if system is ON before performing new calculations
        system_controller = get_controller()
        if system_controller and not system_controller.is_system_on():
            logger.info("System is OFF - returning cached VaR result instead of calculating new one")
            
            # Try to return cached VaR result
            cache_key = f"var_result_{confidence_level}_{time_horizon}_{method}"
            cached_result = system_controller.get_cached_value(cache_key)
            if cached_result:
                logger.debug("Returning cached VaR result while system is OFF")
                return cached_result
            
            # If no cached result, return the last calculated result
            if self.var_history:
                logger.debug("Returning last VaR result from history while system is OFF")
                return self.var_history[-1]
            
            logger.warning("No cached or historical VaR result available while system is OFF")
            return None
        
        start_time = datetime.now()
        
        if not self.positions or self.portfolio_value == 0:
            return None
        
        try:
            if method == "parametric":
                result = await self._calculate_parametric_var(confidence_level, time_horizon)
            elif method == "historical":
                result = await self._calculate_historical_var(confidence_level, time_horizon)
            elif method == "monte_carlo":
                result = await self._calculate_monte_carlo_var(confidence_level, time_horizon)
            elif method == "copula":
                result = await self._calculate_copula_var(confidence_level, time_horizon)
            else:
                logger.error("Unknown VaR method", method=method)
                return None
            
            # Enhanced VaR with copula modeling (if enabled)
            if self.enable_copula_modeling and method != "copula" and len(self.positions) >= 2:
                copula_result = await self._calculate_copula_var(confidence_level, time_horizon)
                if copula_result:
                    # Combine traditional and copula VaR using weighted average
                    copula_weight = 0.3  # 30% weight to copula model
                    traditional_weight = 0.7  # 70% weight to traditional model
                    
                    combined_var = (
                        traditional_weight * result.portfolio_var +
                        copula_weight * copula_result.portfolio_var
                    )
                    
                    result.portfolio_var = combined_var
                    result.calculation_method = f"{method}_copula_enhanced"
            
            # Apply regime adjustment
            regime_multiplier = self.regime_multipliers.get(
                self.correlation_tracker.current_regime, 1.0
            )
            
            if result:
                result.portfolio_var *= regime_multiplier
                for symbol in result.component_vars:
                    result.component_vars[symbol] *= regime_multiplier
                    result.marginal_vars[symbol] *= regime_multiplier
                
                result.correlation_regime = self.correlation_tracker.current_regime.value
                result.performance_ms = (datetime.now() - start_time).total_seconds() * 1000
                
                # Calculate Enhanced Risk Metrics if enabled
                if calculate_expected_shortfall and self.calculate_expected_shortfall:
                    await self._enhance_var_with_es_metrics(result, method, confidence_level, time_horizon)
                
                # Cache the result for when system is OFF
                if system_controller:
                    cache_key = f"var_result_{confidence_level}_{time_horizon}_{method}"
                    system_controller.cache_value(cache_key, result, ttl_seconds=300)  # Cache for 5 minutes
                    logger.debug("Cached VaR result for OFF-system access")
            
            return result
        
        except Exception as e:
            logger.error("VaR calculation failed", error=str(e), exc_info=True)
            
            # Handle VaR calculation failure with proper error context
            from src.core.errors.base_exceptions import SystemError, ErrorContext
            from src.core.errors.error_handler import get_error_handler
            
            error_handler = get_error_handler()
            context = ErrorContext(
                additional_data={
                    "confidence_level": confidence_level,
                    "time_horizon": time_horizon,
                    "method": method,
                    "portfolio_value": self.portfolio_value,
                    "position_count": len(self.positions)
                }
            )
            
            # Create specific error for VaR calculation failure
            var_error = SystemError(
                message=f"VaR calculation failed using {method} method: {str(e)}",
                context=context,
                cause=e,
                recoverable=True
            )
            
            # Register calculate_var as mandatory response function
            error_handler.register_mandatory_response_function(
                "calculate_var",
                validator=lambda result: result is not None and hasattr(result, 'portfolio_var') and result.portfolio_var > 0
            )
            
            # Handle the error and return None if no recovery possible
            result = error_handler.handle_exception(var_error, context, function_name="calculate_var")
            return result
    
    async def _calculate_copula_var(
        self,
        confidence_level: float,
        time_horizon: int
    ) -> Optional[VaRResult]:
        """Calculate VaR using dynamic copula modeling"""
        
        if len(self.positions) < 2:
            return None
        
        try:
            # Get top 2 positions for pairwise copula analysis
            sorted_positions = sorted(
                self.positions.items(),
                key=lambda x: abs(x[1].market_value),
                reverse=True
            )
            
            if len(sorted_positions) < 2:
                return None
            
            symbol1, pos1 = sorted_positions[0]
            symbol2, pos2 = sorted_positions[1]
            
            # Get historical returns for both assets
            returns1 = self._get_asset_returns(symbol1)
            returns2 = self._get_asset_returns(symbol2)
            
            if len(returns1) < 100 or len(returns2) < 100:
                return None  # Insufficient data
            
            # Calculate portfolio weights
            total_value = pos1.market_value + pos2.market_value
            weights = np.array([
                pos1.market_value / total_value,
                pos2.market_value / total_value
            ])
            
            # Calculate copula VaR
            copula_result = self.copula_calculator.calculate_copula_var(
                returns1=returns1,
                returns2=returns2,
                weights=weights,
                n_simulations=50000,  # Reduced for performance
                confidence_level=confidence_level
            )
            
            # Scale for time horizon
            time_scaling = np.sqrt(time_horizon)
            portfolio_var = copula_result.var_estimate * total_value * time_scaling
            
            # Calculate component VaRs (simplified)
            component_vars = {
                symbol1: portfolio_var * weights[0],
                symbol2: portfolio_var * weights[1]
            }
            
            # Add other positions with simplified calculation
            for symbol, position in self.positions.items():
                if symbol not in component_vars:
                    # Use position volatility for individual VaR
                    z_score = stats.norm.ppf(confidence_level)
                    individual_var = position.market_value * position.volatility * z_score * time_scaling
                    component_vars[symbol] = individual_var
            
            marginal_vars = {}
            for symbol, component_var in component_vars.items():
                weight = self.positions[symbol].market_value / self.portfolio_value
                marginal_vars[symbol] = component_var / weight if weight > 0 else 0
            
            return VaRResult(
                timestamp=datetime.now(),
                confidence_level=confidence_level,
                time_horizon_days=time_horizon,
                portfolio_var=portfolio_var,
                component_vars=component_vars,
                marginal_vars=marginal_vars,
                correlation_regime=f"{copula_result.regime.value}_copula",
                calculation_method="copula",
                performance_ms=copula_result.calculation_time_ms,
                expected_shortfall=None,
                component_es=None,
                tail_risk_metrics=None,
                risk_attribution=None
            )
            
        except Exception as e:
            logger.error(f"Copula VaR calculation failed: {e}")
            return None
    
    def _get_asset_returns(self, symbol: str) -> np.ndarray:
        """Get historical returns for an asset"""
        if symbol in self.correlation_tracker.asset_returns:
            returns_list = list(self.correlation_tracker.asset_returns[symbol])
            if len(returns_list) > 0:
                return np.array([ret[1] for ret in returns_list])  # Extract return values
        return np.array([])
    
    async def calculate_tail_risk_metrics(self) -> Dict:
        """Calculate comprehensive tail risk metrics using copula models"""
        
        if not self.enable_copula_modeling or len(self.positions) < 2:
            return {}
        
        try:
            # Get top 2 positions
            sorted_positions = sorted(
                self.positions.items(),
                key=lambda x: abs(x[1].market_value),
                reverse=True
            )
            
            if len(sorted_positions) < 2:
                return {}
            
            symbol1, pos1 = sorted_positions[0]
            symbol2, pos2 = sorted_positions[1]
            
            returns1 = self._get_asset_returns(symbol1)
            returns2 = self._get_asset_returns(symbol2)
            
            if len(returns1) < 100 or len(returns2) < 100:
                return {}
            
            # Calculate portfolio weights
            total_value = pos1.market_value + pos2.market_value
            weights = np.array([
                pos1.market_value / total_value,
                pos2.market_value / total_value
            ])
            
            # Calculate comprehensive tail risk metrics
            tail_metrics = self.copula_calculator.calculate_tail_risk_metrics(
                returns1=returns1,
                returns2=returns2,
                weights=weights
            )
            
            return {
                'tail_dependency_lower': tail_metrics.get('tail_dependency_lower', 0),
                'tail_dependency_upper': tail_metrics.get('tail_dependency_upper', 0),
                'copula_type': tail_metrics.get('copula_type', 'unknown'),
                'market_regime': tail_metrics.get('regime', 'unknown'),
                'model_selection_aic': tail_metrics.get('model_selection_aic', 0),
                'var_estimates': tail_metrics.get('var_estimates', {}),
                'asset_pair': f"{symbol1}_{symbol2}"
            }
            
        except Exception as e:
            logger.error(f"Tail risk metrics calculation failed: {e}")
            return {}
    
    def get_copula_summary(self) -> Dict:
        """Get summary of copula modeling results"""
        if not self.enable_copula_modeling:
            return {'copula_modeling': 'disabled'}
        
        return {
            'copula_modeling': 'enabled',
            'copula_calculator': str(type(self.copula_calculator).__name__),
            'confidence_levels': self.confidence_levels,
            'cache_size': len(self.copula_cache)
        }
    
    async def _calculate_parametric_var(
        self, 
        confidence_level: float, 
        time_horizon: int
    ) -> VaRResult:
        """Calculate parametric VaR using correlation matrix"""
        
        # Get correlation matrix
        correlation_matrix = self.correlation_tracker.get_correlation_matrix()
        if correlation_matrix is None:
            # Fallback to uncorrelated calculation
            correlation_matrix = np.eye(len(self.positions))
        
        # Build portfolio vectors
        symbols = list(self.positions.keys())
        weights = np.array([
            self.positions[symbol].market_value / self.portfolio_value 
            for symbol in symbols
        ])
        
        volatilities = np.array([
            self.positions[symbol].volatility for symbol in symbols
        ])
        
        # Ensure correlation matrix matches positions
        if correlation_matrix.shape[0] != len(symbols):
            # Build correlation matrix from available data or use identity
            n_assets = len(symbols)
            correlation_matrix = np.eye(n_assets)
            
            # Try to map existing correlations
            asset_map = self.correlation_tracker.asset_index
            for i, symbol1 in enumerate(symbols):
                for j, symbol2 in enumerate(symbols):
                    if symbol1 in asset_map and symbol2 in asset_map:
                        idx1, idx2 = asset_map[symbol1], asset_map[symbol2]
                        if (idx1 < self.correlation_tracker.correlation_matrix.shape[0] and 
                            idx2 < self.correlation_tracker.correlation_matrix.shape[1]):
                            correlation_matrix[i, j] = self.correlation_tracker.correlation_matrix[idx1, idx2]
        
        # Calculate portfolio volatility
        # σ_p = sqrt(w^T * Σ * w)
        # where Σ = D * R * D (D = diag(volatilities), R = correlation matrix)
        
        volatility_matrix = np.outer(volatilities, volatilities) * correlation_matrix
        portfolio_variance = np.dot(weights, np.dot(volatility_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Scale for time horizon
        time_scaling = np.sqrt(time_horizon)
        portfolio_volatility_scaled = portfolio_volatility * time_scaling
        
        # Calculate VaR
        z_score = stats.norm.ppf(confidence_level)
        portfolio_var = self.portfolio_value * portfolio_volatility_scaled * z_score
        
        # Calculate component VaRs (contribution to total VaR)
        component_vars = {}
        marginal_vars = {}
        
        for i, symbol in enumerate(symbols):
            # Component VaR = weight * marginal VaR
            marginal_contribution = (
                weights[i] * volatilities[i] * 
                np.sum(correlation_matrix[i, :] * weights * volatilities) /
                portfolio_variance
            )
            
            component_var = marginal_contribution * portfolio_var
            marginal_var = component_var / weights[i] if weights[i] > 0 else 0
            
            component_vars[symbol] = component_var
            marginal_vars[symbol] = marginal_var
        
        return VaRResult(
            timestamp=datetime.now(),
            confidence_level=confidence_level,
            time_horizon_days=time_horizon,
            portfolio_var=portfolio_var,
            component_vars=component_vars,
            marginal_vars=marginal_vars,
            correlation_regime="",  # Will be set by caller
            calculation_method="parametric",
            performance_ms=0,  # Will be set by caller
            expected_shortfall=None,  # Will be calculated later if enabled
            component_es=None,
            tail_risk_metrics=None,
            risk_attribution=None
        )
    
    async def _calculate_historical_var(
        self, 
        confidence_level: float, 
        time_horizon: int
    ) -> VaRResult:
        """Calculate historical VaR using historical returns"""
        
        # Get historical returns for all positions
        returns_data = {}
        min_observations = 252  # 1 year minimum
        
        for symbol in self.positions:
            if symbol in self.correlation_tracker.asset_returns:
                returns = [ret[1] for ret in self.correlation_tracker.asset_returns[symbol]]
                if len(returns) >= min_observations:
                    returns_data[symbol] = returns[-min_observations:]  # Use recent data
        
        if not returns_data:
            # Fallback to parametric
            return await self._calculate_parametric_var(confidence_level, time_horizon)
        
        # Calculate historical portfolio returns
        portfolio_returns = []
        min_length = min(len(returns) for returns in returns_data.values())
        
        for t in range(min_length):
            portfolio_return = 0.0
            for symbol in returns_data:
                weight = self.positions[symbol].market_value / self.portfolio_value
                portfolio_return += weight * returns_data[symbol][t]
            portfolio_returns.append(portfolio_return)
        
        # Scale for time horizon
        portfolio_returns = np.array(portfolio_returns) * np.sqrt(time_horizon)
        
        # Calculate VaR as percentile
        var_percentile = (1 - confidence_level) * 100
        return_var = np.percentile(portfolio_returns, var_percentile)
        portfolio_var = abs(return_var * self.portfolio_value)
        
        # Calculate component contributions (simplified)
        component_vars = {}
        marginal_vars = {}
        
        for symbol in self.positions:
            weight = self.positions[symbol].market_value / self.portfolio_value
            component_vars[symbol] = portfolio_var * weight
            marginal_vars[symbol] = portfolio_var  # Simplified
        
        return VaRResult(
            timestamp=datetime.now(),
            confidence_level=confidence_level,
            time_horizon_days=time_horizon,
            portfolio_var=portfolio_var,
            component_vars=component_vars,
            marginal_vars=marginal_vars,
            correlation_regime="",
            calculation_method="historical",
            performance_ms=0,
            expected_shortfall=None,
            component_es=None,
            tail_risk_metrics=None,
            risk_attribution=None
        )
    
    async def _calculate_monte_carlo_var(
        self, 
        confidence_level: float, 
        time_horizon: int,
        n_simulations: int = 10000
    ) -> VaRResult:
        """Calculate Monte Carlo VaR"""
        
        # Get correlation matrix and position data
        correlation_matrix = self.correlation_tracker.get_correlation_matrix()
        if correlation_matrix is None:
            return await self._calculate_parametric_var(confidence_level, time_horizon)
        
        symbols = list(self.positions.keys())
        weights = np.array([
            self.positions[symbol].market_value / self.portfolio_value 
            for symbol in symbols
        ])
        volatilities = np.array([
            self.positions[symbol].volatility for symbol in symbols
        ])
        
        # Generate correlated random returns
        mean_returns = np.zeros(len(symbols))  # Assume zero mean for VaR
        cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix
        
        # Scale for time horizon
        cov_matrix *= time_horizon
        
        # Monte Carlo simulation
        np.random.seed(42)  # For reproducibility
        random_returns = np.random.multivariate_normal(
            mean_returns, cov_matrix, n_simulations
        )
        
        # Calculate portfolio returns for each simulation
        portfolio_returns = np.dot(random_returns, weights)
        
        # Calculate VaR
        var_percentile = (1 - confidence_level) * 100
        return_var = np.percentile(portfolio_returns, var_percentile)
        portfolio_var = abs(return_var * self.portfolio_value)
        
        # Component analysis using regression
        component_vars = {}
        marginal_vars = {}
        
        for i, symbol in enumerate(symbols):
            # Marginal contribution via regression
            asset_contributions = random_returns[:, i] * weights[i] * self.portfolio_value
            correlation_with_portfolio = np.corrcoef(asset_contributions, portfolio_returns * self.portfolio_value)[0, 1]
            
            component_var = portfolio_var * weights[i] * correlation_with_portfolio
            marginal_var = component_var / weights[i] if weights[i] > 0 else 0
            
            component_vars[symbol] = component_var
            marginal_vars[symbol] = marginal_var
        
        return VaRResult(
            timestamp=datetime.now(),
            confidence_level=confidence_level,
            time_horizon_days=time_horizon,
            portfolio_var=portfolio_var,
            component_vars=component_vars,
            marginal_vars=marginal_vars,
            correlation_regime="",
            calculation_method="monte_carlo",
            performance_ms=0,
            expected_shortfall=None,
            component_es=None,
            tail_risk_metrics=None,
            risk_attribution=None
        )
    
    def _check_var_breach(self, var_result: VaRResult):
        """Check for VaR limit breaches"""
        # Define VaR limits as percentage of portfolio value
        var_limits = {
            0.95: 0.02,  # 2% daily VaR at 95% confidence
            0.99: 0.05   # 5% daily VaR at 99% confidence
        }
        
        var_percentage = var_result.portfolio_var / self.portfolio_value
        limit = var_limits.get(var_result.confidence_level, 0.05)
        
        if var_percentage > limit:
            self.event_bus.publish(
                self.event_bus.create_event(
                    EventType.RISK_BREACH,
                    {
                        'type': 'VAR_LIMIT_BREACH',
                        'var_percentage': var_percentage,
                        'limit': limit,
                        'portfolio_value': self.portfolio_value,
                        'var_amount': var_result.portfolio_var,
                        'confidence_level': var_result.confidence_level
                    },
                    'VaRCalculator'
                )
            )
            
            logger.warning("VaR limit breach detected",
                          var_percentage=f"{var_percentage:.2%}",
                          limit=f"{limit:.2%}",
                          confidence_level=var_result.confidence_level)
    
    def get_performance_stats(self) -> Dict:
        """Get VaR calculation performance statistics"""
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
    
    def get_latest_var(self, confidence_level: float = 0.95) -> Optional[VaRResult]:
        """Get latest VaR result for specified confidence level"""
        for var_result in reversed(self.var_history):
            if var_result.confidence_level == confidence_level:
                return var_result
        return None
    
    def get_var_summary(self) -> Dict:
        """Get comprehensive VaR summary"""
        latest_var = self.get_latest_var()
        
        if not latest_var:
            return {"status": "No VaR calculated"}
        
        summary = {
            "timestamp": latest_var.timestamp.isoformat(),
            "portfolio_value": self.portfolio_value,
            "portfolio_var": latest_var.portfolio_var,
            "var_percentage": latest_var.portfolio_var / self.portfolio_value,
            "confidence_level": latest_var.confidence_level,
            "time_horizon_days": latest_var.time_horizon_days,
            "correlation_regime": latest_var.correlation_regime,
            "calculation_method": latest_var.calculation_method,
            "performance_ms": latest_var.performance_ms,
            "top_contributors": dict(sorted(
                latest_var.component_vars.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:5])  # Top 5 risk contributors
        }
        
        # Add copula modeling summary
        copula_summary = self.get_copula_summary()
        summary.update(copula_summary)
        
        return summary
    
    async def _enhance_var_with_es_metrics(
        self,
        var_result: VaRResult,
        method: str,
        confidence_level: float,
        time_horizon: int
    ) -> None:
        """
        Enhance VaR result with Expected Shortfall and tail risk metrics.
        
        Args:
            var_result: VaRResult to enhance
            method: VaR calculation method used
            confidence_level: Confidence level
            time_horizon: Time horizon in days
        """
        
        if method == "historical":
            await self._calculate_historical_es(var_result, confidence_level, time_horizon)
        elif method == "monte_carlo":
            await self._calculate_monte_carlo_es(var_result, confidence_level, time_horizon)
        elif method == "parametric":
            await self._calculate_parametric_es(var_result, confidence_level, time_horizon)
        
        # Calculate risk attribution
        if self.calculate_risk_attribution:
            self._calculate_risk_attribution(var_result)
        
        # Calculate tail risk metrics
        self._calculate_tail_risk_metrics(var_result, confidence_level)
    
    async def _calculate_parametric_es(
        self,
        var_result: VaRResult,
        confidence_level: float,
        time_horizon: int
    ) -> None:
        """
        Calculate Expected Shortfall using parametric method.
        
        For normal distribution: ES = μ + σ * φ(Φ^(-1)(α)) / (1-α)
        where φ is PDF and Φ^(-1) is inverse CDF
        """
        
        # Get portfolio volatility from current correlation matrix
        correlation_matrix = self.correlation_tracker.get_correlation_matrix()
        if correlation_matrix is None:
            correlation_matrix = np.eye(len(self.positions))
        
        symbols = list(self.positions.keys())
        weights = np.array([
            self.positions[symbol].market_value / self.portfolio_value 
            for symbol in symbols
        ])
        
        volatilities = np.array([
            self.positions[symbol].volatility for symbol in symbols
        ])
        
        # Calculate portfolio volatility
        volatility_matrix = np.outer(volatilities, volatilities) * correlation_matrix
        portfolio_variance = np.dot(weights, np.dot(volatility_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Scale for time horizon
        time_scaling = np.sqrt(time_horizon)
        portfolio_volatility_scaled = portfolio_volatility * time_scaling
        
        # Calculate Expected Shortfall for normal distribution
        alpha = 1 - confidence_level
        z_alpha = stats.norm.ppf(confidence_level)
        phi_z_alpha = stats.norm.pdf(z_alpha)
        
        # ES = VaR + σ * φ(z_α) / α
        es_multiplier = phi_z_alpha / alpha
        portfolio_es = var_result.portfolio_var + (self.portfolio_value * portfolio_volatility_scaled * es_multiplier)
        
        var_result.expected_shortfall = portfolio_es
        
        # Calculate component ES
        component_es = {}
        for i, symbol in enumerate(symbols):
            # Component ES proportional to component VaR
            component_weight = var_result.component_vars[symbol] / var_result.portfolio_var
            component_es[symbol] = portfolio_es * component_weight
        
        var_result.component_es = component_es
    
    async def _calculate_historical_es(
        self,
        var_result: VaRResult,
        confidence_level: float,
        time_horizon: int
    ) -> None:
        """
        Calculate Expected Shortfall using historical method.
        
        ES is the average of losses beyond VaR threshold.
        """
        
        # Get historical returns for all positions
        returns_data = {}
        min_observations = 252
        
        for symbol in self.positions:
            if symbol in self.correlation_tracker.asset_returns:
                returns = [ret[1] for ret in self.correlation_tracker.asset_returns[symbol]]
                if len(returns) >= min_observations:
                    returns_data[symbol] = returns[-min_observations:]
        
        if not returns_data:
            # Fallback to parametric ES
            await self._calculate_parametric_es(var_result, confidence_level, time_horizon)
            return
        
        # Calculate historical portfolio returns
        portfolio_returns = []
        min_length = min(len(returns) for returns in returns_data.values())
        
        for t in range(min_length):
            portfolio_return = 0.0
            for symbol in returns_data:
                weight = self.positions[symbol].market_value / self.portfolio_value
                portfolio_return += weight * returns_data[symbol][t]
            portfolio_returns.append(portfolio_return)
        
        # Scale for time horizon
        portfolio_returns = np.array(portfolio_returns) * np.sqrt(time_horizon)
        
        # Calculate VaR percentile
        var_percentile = (1 - confidence_level) * 100
        var_threshold = np.percentile(portfolio_returns, var_percentile)
        
        # Calculate Expected Shortfall as average of tail losses
        tail_losses = portfolio_returns[portfolio_returns <= var_threshold]
        if len(tail_losses) > 0:
            expected_shortfall = abs(np.mean(tail_losses) * self.portfolio_value)
        else:
            expected_shortfall = var_result.portfolio_var * 1.5  # Conservative estimate
        
        var_result.expected_shortfall = expected_shortfall
        
        # Calculate component ES (simplified)
        component_es = {}
        for symbol in self.positions:
            weight = self.positions[symbol].market_value / self.portfolio_value
            component_es[symbol] = expected_shortfall * weight
        
        var_result.component_es = component_es
    
    async def _calculate_monte_carlo_es(
        self,
        var_result: VaRResult,
        confidence_level: float,
        time_horizon: int
    ) -> None:
        """
        Calculate Expected Shortfall using Monte Carlo simulation.
        """
        
        correlation_matrix = self.correlation_tracker.get_correlation_matrix()
        if correlation_matrix is None:
            await self._calculate_parametric_es(var_result, confidence_level, time_horizon)
            return
        
        symbols = list(self.positions.keys())
        weights = np.array([
            self.positions[symbol].market_value / self.portfolio_value 
            for symbol in symbols
        ])
        volatilities = np.array([
            self.positions[symbol].volatility for symbol in symbols
        ])
        
        # Generate correlated random returns
        n_simulations = 10000
        mean_returns = np.zeros(len(symbols))
        cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix
        cov_matrix *= time_horizon
        
        np.random.seed(42)
        random_returns = np.random.multivariate_normal(
            mean_returns, cov_matrix, n_simulations
        )
        
        # Calculate portfolio returns
        portfolio_returns = np.dot(random_returns, weights)
        
        # Calculate VaR threshold
        var_percentile = (1 - confidence_level) * 100
        var_threshold = np.percentile(portfolio_returns, var_percentile)
        
        # Calculate Expected Shortfall
        tail_losses = portfolio_returns[portfolio_returns <= var_threshold]
        if len(tail_losses) > 0:
            expected_shortfall = abs(np.mean(tail_losses) * self.portfolio_value)
        else:
            expected_shortfall = var_result.portfolio_var * 1.5
        
        var_result.expected_shortfall = expected_shortfall
        
        # Calculate component ES using simulation
        component_es = {}
        for i, symbol in enumerate(symbols):
            # Component contributions in tail scenarios
            asset_contributions = random_returns[:, i] * weights[i] * self.portfolio_value
            tail_contributions = asset_contributions[portfolio_returns <= var_threshold]
            
            if len(tail_contributions) > 0:
                component_es[symbol] = abs(np.mean(tail_contributions))
            else:
                component_es[symbol] = var_result.component_vars[symbol] * 1.5
        
        var_result.component_es = component_es
    
    def _calculate_risk_attribution(self, var_result: VaRResult) -> None:
        """
        Calculate risk attribution metrics for portfolio components.
        
        Risk attribution shows how much each position contributes to total risk.
        """
        
        risk_attribution = {}
        total_var = var_result.portfolio_var
        
        for symbol, component_var in var_result.component_vars.items():
            # Attribution as percentage of total VaR
            attribution_pct = (component_var / total_var) if total_var > 0 else 0
            
            # Risk-adjusted attribution (position size vs risk contribution)
            position_weight = self.positions[symbol].market_value / self.portfolio_value
            risk_efficiency = attribution_pct / position_weight if position_weight > 0 else 0
            
            risk_attribution[symbol] = {
                'var_contribution': component_var,
                'var_contribution_pct': attribution_pct,
                'position_weight': position_weight,
                'risk_efficiency': risk_efficiency,
                'excess_risk': attribution_pct - position_weight
            }
        
        var_result.risk_attribution = risk_attribution
    
    def _calculate_tail_risk_metrics(self, var_result: VaRResult, confidence_level: float) -> None:
        """
        Calculate comprehensive tail risk metrics.
        """
        
        tail_metrics = {}
        
        # Basic tail risk ratios
        if var_result.expected_shortfall and var_result.portfolio_var > 0:
            tail_metrics['es_var_ratio'] = var_result.expected_shortfall / var_result.portfolio_var
        else:
            tail_metrics['es_var_ratio'] = 1.0
        
        # Tail risk concentration
        if var_result.component_es and var_result.expected_shortfall:
            max_component_es = max(var_result.component_es.values())
            tail_metrics['tail_concentration'] = max_component_es / var_result.expected_shortfall
        else:
            tail_metrics['tail_concentration'] = 0.0
        
        # Risk-adjusted metrics
        tail_metrics['var_as_pct_of_portfolio'] = var_result.portfolio_var / self.portfolio_value
        tail_metrics['es_as_pct_of_portfolio'] = (var_result.expected_shortfall / self.portfolio_value) if var_result.expected_shortfall else 0
        
        # Confidence level metrics
        tail_metrics['confidence_level'] = confidence_level
        tail_metrics['tail_probability'] = 1 - confidence_level
        
        var_result.tail_risk_metrics = tail_metrics
    
    def calculate_coherent_risk_measures(self, confidence_level: float = 0.95) -> Dict[str, float]:
        """
        Calculate coherent risk measures for the portfolio.
        
        Coherent risk measures satisfy:
        1. Monotonicity
        2. Subadditivity  
        3. Positive homogeneity
        4. Translation invariance
        
        Returns:
            Dict with coherent risk measures
        """
        
        latest_var = self.get_latest_var(confidence_level)
        if not latest_var or not latest_var.expected_shortfall:
            return {}
        
        # Expected Shortfall is coherent, VaR is not
        coherent_measures = {
            'expected_shortfall': latest_var.expected_shortfall,
            'conditional_var': latest_var.expected_shortfall,  # Same as ES
            'tail_var': latest_var.expected_shortfall,
            'coherent_risk_ratio': latest_var.expected_shortfall / latest_var.portfolio_var if latest_var.portfolio_var > 0 else 1.0
        }
        
        # Calculate spectral risk measures if we have multiple quantiles
        if len(self.tail_risk_quantiles) > 1:
            # Simplified spectral risk measure (equal weights)
            spectral_weights = np.ones(len(self.tail_risk_quantiles)) / len(self.tail_risk_quantiles)
            spectral_risk = sum(w * latest_var.expected_shortfall for w in spectral_weights)
            coherent_measures['spectral_risk'] = spectral_risk
        
        return coherent_measures