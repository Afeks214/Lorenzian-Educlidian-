"""
Advanced Position Sizing Engine with Kelly Criterion and Optimization
=====================================================================

This module implements a sophisticated position sizing system that includes:
- Kelly Criterion with safety constraints
- Volatility-based position sizing
- Dynamic risk adjustment based on market conditions
- Portfolio heat and concentration limits
- Correlation-based position adjustments
- Numba JIT optimized calculations

Author: Risk Management System
Date: 2025-07-17
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
import logging
from numba import jit, njit, prange
from scipy import stats, optimize
import warnings
from enum import Enum

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class PositionSizingMethod(Enum):
    """Position sizing methods"""
    KELLY_CRITERION = "kelly_criterion"
    VOLATILITY_BASED = "volatility_based"
    RISK_PARITY = "risk_parity"
    EQUAL_WEIGHT = "equal_weight"
    DYNAMIC_OPTIMIZATION = "dynamic_optimization"


@dataclass
class PositionSizingConfig:
    """Configuration for position sizing"""
    # Kelly Criterion parameters
    kelly_enabled: bool = True
    kelly_max_fraction: float = 0.25
    kelly_lookback_period: int = 100
    kelly_confidence_threshold: float = 0.6
    kelly_safety_factor: float = 0.5
    
    # Risk parameters
    max_position_size: float = 0.10
    min_position_size: float = 0.005
    max_risk_per_trade: float = 0.02
    max_portfolio_risk: float = 0.15
    
    # Volatility parameters
    volatility_lookback: int = 20
    volatility_multiplier: float = 2.0
    volatility_adjustment: bool = True
    
    # Market condition adjustments
    trending_multiplier: float = 1.2
    ranging_multiplier: float = 0.8
    high_vol_multiplier: float = 0.7
    low_vol_multiplier: float = 1.1
    
    # Correlation parameters
    correlation_adjustment: bool = True
    max_correlation: float = 0.7
    correlation_lookback: int = 60
    correlation_penalty: float = 0.5
    
    # Portfolio constraints
    max_concentration: float = 0.15
    max_sector_exposure: float = 0.30
    max_leverage: float = 3.0
    
    # Performance tracking
    performance_decay_threshold: float = 0.20
    recalibration_threshold: float = 0.15
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'kelly_enabled': self.kelly_enabled,
            'kelly_max_fraction': self.kelly_max_fraction,
            'kelly_lookback_period': self.kelly_lookback_period,
            'kelly_confidence_threshold': self.kelly_confidence_threshold,
            'kelly_safety_factor': self.kelly_safety_factor,
            'max_position_size': self.max_position_size,
            'min_position_size': self.min_position_size,
            'max_risk_per_trade': self.max_risk_per_trade,
            'max_portfolio_risk': self.max_portfolio_risk,
            'volatility_lookback': self.volatility_lookback,
            'volatility_multiplier': self.volatility_multiplier,
            'volatility_adjustment': self.volatility_adjustment,
            'trending_multiplier': self.trending_multiplier,
            'ranging_multiplier': self.ranging_multiplier,
            'high_vol_multiplier': self.high_vol_multiplier,
            'low_vol_multiplier': self.low_vol_multiplier,
            'correlation_adjustment': self.correlation_adjustment,
            'max_correlation': self.max_correlation,
            'correlation_lookback': self.correlation_lookback,
            'correlation_penalty': self.correlation_penalty,
            'max_concentration': self.max_concentration,
            'max_sector_exposure': self.max_sector_exposure,
            'max_leverage': self.max_leverage,
            'performance_decay_threshold': self.performance_decay_threshold,
            'recalibration_threshold': self.recalibration_threshold
        }


@dataclass
class PositionSizingResult:
    """Result of position sizing calculation"""
    position_size: float
    risk_amount: float
    confidence_score: float
    method_used: str
    risk_adjusted_size: float
    volatility_adjustment: float
    correlation_adjustment: float
    kelly_fraction: float
    portfolio_heat: float
    constraints_applied: List[str]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'position_size': self.position_size,
            'risk_amount': self.risk_amount,
            'confidence_score': self.confidence_score,
            'method_used': self.method_used,
            'risk_adjusted_size': self.risk_adjusted_size,
            'volatility_adjustment': self.volatility_adjustment,
            'correlation_adjustment': self.correlation_adjustment,
            'kelly_fraction': self.kelly_fraction,
            'portfolio_heat': self.portfolio_heat,
            'constraints_applied': self.constraints_applied,
            'metadata': self.metadata
        }


@dataclass
class MarketCondition:
    """Market condition data"""
    volatility: float
    trend_strength: float
    market_regime: str
    correlation_level: float
    liquidity_score: float
    
    def is_trending(self) -> bool:
        """Check if market is trending"""
        return self.trend_strength > 0.6
    
    def is_high_volatility(self) -> bool:
        """Check if volatility is high"""
        return self.volatility > 0.25
    
    def is_high_correlation(self) -> bool:
        """Check if correlation is high"""
        return self.correlation_level > 0.7


@dataclass
class TradeOpportunity:
    """Trade opportunity data"""
    symbol: str
    signal_confidence: float
    expected_return: float
    expected_volatility: float
    stop_loss_distance: float
    take_profit_distance: float
    win_probability: float
    historical_performance: Dict[str, float]
    sector: str
    market_cap: float
    liquidity_score: float
    
    def risk_reward_ratio(self) -> float:
        """Calculate risk-reward ratio"""
        if self.stop_loss_distance == 0:
            return 0.0
        return self.take_profit_distance / self.stop_loss_distance
    
    def expected_value(self) -> float:
        """Calculate expected value"""
        win_amount = self.take_profit_distance * self.win_probability
        loss_amount = self.stop_loss_distance * (1 - self.win_probability)
        return win_amount - loss_amount


@dataclass
class PortfolioState:
    """Current portfolio state"""
    total_value: float
    cash_available: float
    positions: Dict[str, float]
    sector_exposures: Dict[str, float]
    current_leverage: float
    current_heat: float
    correlation_matrix: np.ndarray
    sector_correlations: Dict[str, float]
    recent_performance: Dict[str, float]
    
    def get_available_capital(self) -> float:
        """Get available capital for new positions"""
        return self.cash_available
    
    def get_current_exposure(self, symbol: str) -> float:
        """Get current exposure to symbol"""
        return self.positions.get(symbol, 0.0)
    
    def get_sector_exposure(self, sector: str) -> float:
        """Get current sector exposure"""
        return self.sector_exposures.get(sector, 0.0)


# Numba JIT optimized functions
@njit
def calculate_kelly_fraction(
    win_probability: float,
    win_amount: float,
    loss_amount: float,
    safety_factor: float = 0.5
) -> float:
    """
    Calculate Kelly fraction - JIT optimized
    
    Kelly fraction = (bp - q) / b
    where:
    - b = odds received on the wager (win_amount / loss_amount)
    - p = probability of winning
    - q = probability of losing (1 - p)
    """
    if win_probability <= 0.0 or win_probability >= 1.0:
        return 0.0
    
    if loss_amount <= 0.0:
        return 0.0
    
    # Calculate Kelly fraction
    b = win_amount / loss_amount
    p = win_probability
    q = 1.0 - p
    
    kelly_fraction = (b * p - q) / b
    
    # Apply safety factor
    kelly_fraction = max(0.0, kelly_fraction * safety_factor)
    
    return kelly_fraction


@njit
def calculate_volatility_adjustment(
    current_volatility: float,
    historical_volatility: float,
    base_size: float,
    multiplier: float = 2.0
) -> float:
    """Calculate volatility-based position size adjustment - JIT optimized"""
    if historical_volatility <= 0.0:
        return base_size
    
    # Inverse volatility scaling
    vol_ratio = historical_volatility / current_volatility
    adjusted_size = base_size * vol_ratio / multiplier
    
    return max(0.0, adjusted_size)


@njit
def calculate_correlation_penalty(
    new_asset_correlations: np.ndarray,
    portfolio_weights: np.ndarray,
    max_correlation: float = 0.7,
    penalty_factor: float = 0.5
) -> float:
    """Calculate correlation penalty - JIT optimized"""
    if len(new_asset_correlations) == 0 or len(portfolio_weights) == 0:
        return 1.0
    
    # Calculate weighted average correlation
    weighted_correlation = np.sum(new_asset_correlations * portfolio_weights)
    
    # Apply penalty if correlation is too high
    if weighted_correlation > max_correlation:
        excess_correlation = weighted_correlation - max_correlation
        penalty = 1.0 - (excess_correlation * penalty_factor)
        return max(0.1, penalty)
    
    return 1.0


@njit
def calculate_portfolio_heat(
    position_risks: np.ndarray,
    correlation_matrix: np.ndarray
) -> float:
    """Calculate portfolio heat (total risk) - JIT optimized"""
    if len(position_risks) == 0:
        return 0.0
    
    # Portfolio variance = w^T * C * w
    portfolio_variance = np.dot(position_risks, np.dot(correlation_matrix, position_risks))
    
    # Portfolio heat is the square root (volatility)
    portfolio_heat = np.sqrt(max(0.0, portfolio_variance))
    
    return portfolio_heat


@njit
def optimize_position_sizes(
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    max_weights: np.ndarray,
    risk_tolerance: float = 0.15
) -> np.ndarray:
    """
    Optimize position sizes using simplified mean-variance optimization
    JIT optimized version
    """
    n_assets = len(expected_returns)
    
    if n_assets == 0:
        return np.array([])
    
    # Initialize equal weights
    weights = np.ones(n_assets) / n_assets
    
    # Simple iterative optimization
    for _ in range(100):  # Max iterations
        # Calculate portfolio return and risk
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
        
        if portfolio_variance <= 0:
            break
        
        # Calculate gradients
        return_gradient = expected_returns
        risk_gradient = 2 * np.dot(covariance_matrix, weights)
        
        # Combine gradients with risk tolerance
        combined_gradient = return_gradient - risk_tolerance * risk_gradient
        
        # Update weights
        step_size = 0.01
        new_weights = weights + step_size * combined_gradient
        
        # Apply constraints
        new_weights = np.maximum(0.0, new_weights)
        new_weights = np.minimum(max_weights, new_weights)
        
        # Normalize
        weight_sum = np.sum(new_weights)
        if weight_sum > 0:
            new_weights = new_weights / weight_sum
        
        # Check convergence
        if np.sum(np.abs(new_weights - weights)) < 1e-6:
            break
        
        weights = new_weights
    
    return weights


class AdvancedPositionSizer:
    """
    Advanced position sizing engine with Kelly Criterion and optimization
    
    This class implements sophisticated position sizing algorithms including
    Kelly Criterion, volatility-based sizing, correlation adjustments,
    and dynamic optimization.
    """
    
    def __init__(self, config: PositionSizingConfig):
        """
        Initialize the advanced position sizer
        
        Args:
            config: Position sizing configuration
        """
        self.config = config
        
        # Performance tracking
        self.sizing_history: List[PositionSizingResult] = []
        self.performance_metrics: Dict[str, float] = {}
        self.calculation_times: List[float] = []
        
        # Market condition tracking
        self.market_conditions: List[MarketCondition] = []
        
        # Optimization parameters
        self.optimization_cache: Dict[str, Any] = {}
        
        logger.info("AdvancedPositionSizer initialized",
                   extra={
                       'config': config.to_dict()
                   })
    
    async def calculate_position_size(
        self,
        opportunity: TradeOpportunity,
        portfolio_state: PortfolioState,
        market_condition: MarketCondition,
        method: PositionSizingMethod = PositionSizingMethod.DYNAMIC_OPTIMIZATION
    ) -> PositionSizingResult:
        """
        Calculate optimal position size for a trade opportunity
        
        Args:
            opportunity: Trade opportunity data
            portfolio_state: Current portfolio state
            market_condition: Current market conditions
            method: Position sizing method to use
        
        Returns:
            PositionSizingResult with sizing details
        """
        start_time = datetime.now()
        
        try:
            # Initialize result
            constraints_applied = []
            metadata = {}
            
            # Calculate base position size using selected method
            if method == PositionSizingMethod.KELLY_CRITERION:
                base_size = await self._calculate_kelly_size(opportunity, portfolio_state)
                method_name = "kelly_criterion"
            elif method == PositionSizingMethod.VOLATILITY_BASED:
                base_size = await self._calculate_volatility_based_size(opportunity, portfolio_state)
                method_name = "volatility_based"
            elif method == PositionSizingMethod.RISK_PARITY:
                base_size = await self._calculate_risk_parity_size(opportunity, portfolio_state)
                method_name = "risk_parity"
            elif method == PositionSizingMethod.EQUAL_WEIGHT:
                base_size = await self._calculate_equal_weight_size(opportunity, portfolio_state)
                method_name = "equal_weight"
            else:  # DYNAMIC_OPTIMIZATION
                base_size = await self._calculate_optimized_size(opportunity, portfolio_state, market_condition)
                method_name = "dynamic_optimization"
            
            # Apply volatility adjustment
            volatility_adjustment = 1.0
            if self.config.volatility_adjustment:
                volatility_adjustment = calculate_volatility_adjustment(
                    opportunity.expected_volatility,
                    np.mean([mc.volatility for mc in self.market_conditions[-20:]] or [0.2]),
                    1.0,
                    self.config.volatility_multiplier
                )
            
            # Apply correlation adjustment
            correlation_adjustment = 1.0
            if self.config.correlation_adjustment:
                correlation_adjustment = await self._calculate_correlation_adjustment(
                    opportunity, portfolio_state
                )
            
            # Apply market condition adjustments
            market_adjustment = self._calculate_market_adjustment(market_condition)
            
            # Combine all adjustments
            adjusted_size = base_size * volatility_adjustment * correlation_adjustment * market_adjustment
            
            # Apply position size constraints
            adjusted_size = max(self.config.min_position_size, adjusted_size)
            adjusted_size = min(self.config.max_position_size, adjusted_size)
            
            # Check concentration limits
            current_exposure = portfolio_state.get_current_exposure(opportunity.symbol)
            max_additional = self.config.max_concentration - current_exposure
            
            if adjusted_size > max_additional:
                adjusted_size = max_additional
                constraints_applied.append("concentration_limit")
            
            # Check sector exposure limits
            current_sector_exposure = portfolio_state.get_sector_exposure(opportunity.sector)
            max_sector_additional = self.config.max_sector_exposure - current_sector_exposure
            
            if adjusted_size > max_sector_additional:
                adjusted_size = max_sector_additional
                constraints_applied.append("sector_exposure_limit")
            
            # Check portfolio risk limits
            portfolio_heat = await self._calculate_portfolio_heat_impact(
                opportunity, adjusted_size, portfolio_state
            )
            
            if portfolio_heat > self.config.max_portfolio_risk:
                # Scale down position to meet risk limit
                scale_factor = self.config.max_portfolio_risk / portfolio_heat
                adjusted_size *= scale_factor
                constraints_applied.append("portfolio_risk_limit")
            
            # Calculate final risk amount
            risk_amount = adjusted_size * opportunity.stop_loss_distance
            
            # Calculate Kelly fraction for reference
            kelly_fraction = calculate_kelly_fraction(
                opportunity.win_probability,
                opportunity.take_profit_distance,
                opportunity.stop_loss_distance,
                self.config.kelly_safety_factor
            )
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(
                opportunity, portfolio_state, market_condition
            )
            
            # Create result
            result = PositionSizingResult(
                position_size=adjusted_size,
                risk_amount=risk_amount,
                confidence_score=confidence_score,
                method_used=method_name,
                risk_adjusted_size=adjusted_size,
                volatility_adjustment=volatility_adjustment,
                correlation_adjustment=correlation_adjustment,
                kelly_fraction=kelly_fraction,
                portfolio_heat=portfolio_heat,
                constraints_applied=constraints_applied,
                metadata=metadata
            )
            
            # Store in history
            self.sizing_history.append(result)
            
            # Keep only recent history
            if len(self.sizing_history) > 10000:
                self.sizing_history = self.sizing_history[-10000:]
            
            return result
            
        except Exception as e:
            logger.error(f"Position sizing calculation failed: {e}")
            # Return minimal position size
            return PositionSizingResult(
                position_size=self.config.min_position_size,
                risk_amount=self.config.min_position_size * opportunity.stop_loss_distance,
                confidence_score=0.0,
                method_used="error_fallback",
                risk_adjusted_size=self.config.min_position_size,
                volatility_adjustment=1.0,
                correlation_adjustment=1.0,
                kelly_fraction=0.0,
                portfolio_heat=0.0,
                constraints_applied=["error_fallback"],
                metadata={"error": str(e)}
            )
        
        finally:
            # Track performance
            calc_time = (datetime.now() - start_time).total_seconds() * 1000
            self.calculation_times.append(calc_time)
            
            if len(self.calculation_times) > 1000:
                self.calculation_times = self.calculation_times[-1000:]
    
    async def _calculate_kelly_size(
        self,
        opportunity: TradeOpportunity,
        portfolio_state: PortfolioState
    ) -> float:
        """Calculate Kelly criterion-based position size"""
        
        # Get historical performance data
        historical_data = opportunity.historical_performance
        
        if not historical_data or len(historical_data) < self.config.kelly_lookback_period:
            # Not enough data, use signal confidence
            kelly_fraction = calculate_kelly_fraction(
                opportunity.win_probability,
                opportunity.take_profit_distance,
                opportunity.stop_loss_distance,
                self.config.kelly_safety_factor
            )
        else:
            # Calculate from historical data
            wins = historical_data.get('wins', 0)
            losses = historical_data.get('losses', 0)
            avg_win = historical_data.get('avg_win', opportunity.take_profit_distance)
            avg_loss = historical_data.get('avg_loss', opportunity.stop_loss_distance)
            
            total_trades = wins + losses
            if total_trades == 0:
                return self.config.min_position_size
            
            win_probability = wins / total_trades
            
            kelly_fraction = calculate_kelly_fraction(
                win_probability,
                avg_win,
                avg_loss,
                self.config.kelly_safety_factor
            )
        
        # Apply confidence threshold
        if opportunity.signal_confidence < self.config.kelly_confidence_threshold:
            kelly_fraction *= opportunity.signal_confidence / self.config.kelly_confidence_threshold
        
        # Apply maximum Kelly fraction limit
        kelly_fraction = min(kelly_fraction, self.config.kelly_max_fraction)
        
        return kelly_fraction
    
    async def _calculate_volatility_based_size(
        self,
        opportunity: TradeOpportunity,
        portfolio_state: PortfolioState
    ) -> float:
        """Calculate volatility-based position size"""
        
        # Target volatility approach
        target_volatility = self.config.max_risk_per_trade
        
        # Calculate position size based on volatility
        if opportunity.expected_volatility > 0:
            base_size = target_volatility / opportunity.expected_volatility
        else:
            base_size = self.config.min_position_size
        
        # Apply volatility multiplier
        adjusted_size = base_size / self.config.volatility_multiplier
        
        return adjusted_size
    
    async def _calculate_risk_parity_size(
        self,
        opportunity: TradeOpportunity,
        portfolio_state: PortfolioState
    ) -> float:
        """Calculate risk parity-based position size"""
        
        # Equal risk contribution approach
        target_risk = self.config.max_risk_per_trade
        
        # Calculate position size to achieve target risk
        position_risk = opportunity.expected_volatility * opportunity.stop_loss_distance
        
        if position_risk > 0:
            size = target_risk / position_risk
        else:
            size = self.config.min_position_size
        
        return size
    
    async def _calculate_equal_weight_size(
        self,
        opportunity: TradeOpportunity,
        portfolio_state: PortfolioState
    ) -> float:
        """Calculate equal weight position size"""
        
        # Simple equal weight approach
        n_positions = len(portfolio_state.positions) + 1
        
        if n_positions == 0:
            return self.config.min_position_size
        
        equal_weight = 1.0 / n_positions
        
        return equal_weight
    
    async def _calculate_optimized_size(
        self,
        opportunity: TradeOpportunity,
        portfolio_state: PortfolioState,
        market_condition: MarketCondition
    ) -> float:
        """Calculate optimized position size using multiple methods"""
        
        # Calculate sizes using different methods
        kelly_size = await self._calculate_kelly_size(opportunity, portfolio_state)
        vol_size = await self._calculate_volatility_based_size(opportunity, portfolio_state)
        risk_parity_size = await self._calculate_risk_parity_size(opportunity, portfolio_state)
        
        # Weight the methods based on market conditions and opportunity quality
        weights = self._calculate_method_weights(opportunity, market_condition)
        
        # Combine methods
        combined_size = (
            weights['kelly'] * kelly_size +
            weights['volatility'] * vol_size +
            weights['risk_parity'] * risk_parity_size
        )
        
        return combined_size
    
    def _calculate_method_weights(
        self,
        opportunity: TradeOpportunity,
        market_condition: MarketCondition
    ) -> Dict[str, float]:
        """Calculate weights for different sizing methods"""
        
        weights = {'kelly': 0.4, 'volatility': 0.3, 'risk_parity': 0.3}
        
        # Adjust weights based on signal confidence
        if opportunity.signal_confidence > 0.8:
            weights['kelly'] += 0.2
            weights['volatility'] -= 0.1
            weights['risk_parity'] -= 0.1
        
        # Adjust weights based on market conditions
        if market_condition.is_high_volatility():
            weights['volatility'] += 0.2
            weights['kelly'] -= 0.1
            weights['risk_parity'] -= 0.1
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    async def _calculate_correlation_adjustment(
        self,
        opportunity: TradeOpportunity,
        portfolio_state: PortfolioState
    ) -> float:
        """Calculate correlation-based adjustment"""
        
        if len(portfolio_state.positions) == 0:
            return 1.0
        
        # Get correlations with existing positions
        correlations = []
        weights = []
        
        for symbol, weight in portfolio_state.positions.items():
            if symbol in portfolio_state.correlation_matrix:
                # This is simplified - in reality, you'd need proper correlation data
                correlation = portfolio_state.sector_correlations.get(opportunity.sector, 0.0)
                correlations.append(correlation)
                weights.append(weight)
        
        if not correlations:
            return 1.0
        
        # Calculate adjustment using numba optimized function
        correlations_array = np.array(correlations)
        weights_array = np.array(weights)
        
        adjustment = calculate_correlation_penalty(
            correlations_array,
            weights_array,
            self.config.max_correlation,
            self.config.correlation_penalty
        )
        
        return adjustment
    
    def _calculate_market_adjustment(self, market_condition: MarketCondition) -> float:
        """Calculate market condition-based adjustment"""
        
        adjustment = 1.0
        
        # Trend adjustment
        if market_condition.is_trending():
            adjustment *= self.config.trending_multiplier
        else:
            adjustment *= self.config.ranging_multiplier
        
        # Volatility adjustment
        if market_condition.is_high_volatility():
            adjustment *= self.config.high_vol_multiplier
        else:
            adjustment *= self.config.low_vol_multiplier
        
        return adjustment
    
    async def _calculate_portfolio_heat_impact(
        self,
        opportunity: TradeOpportunity,
        position_size: float,
        portfolio_state: PortfolioState
    ) -> float:
        """Calculate portfolio heat impact of new position"""
        
        # Current portfolio risks
        current_risks = []
        for symbol, weight in portfolio_state.positions.items():
            # Estimate risk (simplified)
            risk = weight * 0.2  # Assume 20% volatility
            current_risks.append(risk)
        
        # Add new position risk
        new_risk = position_size * opportunity.expected_volatility
        current_risks.append(new_risk)
        
        # Calculate portfolio heat using correlation matrix
        if len(current_risks) > 0:
            risks_array = np.array(current_risks)
            
            # Create correlation matrix (simplified)
            n = len(risks_array)
            correlation_matrix = np.eye(n)
            
            # Fill with estimated correlations
            for i in range(n):
                for j in range(n):
                    if i != j:
                        correlation_matrix[i, j] = 0.3  # Assume moderate correlation
            
            portfolio_heat = calculate_portfolio_heat(risks_array, correlation_matrix)
        else:
            portfolio_heat = new_risk
        
        return portfolio_heat
    
    def _calculate_confidence_score(
        self,
        opportunity: TradeOpportunity,
        portfolio_state: PortfolioState,
        market_condition: MarketCondition
    ) -> float:
        """Calculate confidence score for position sizing"""
        
        # Base confidence from signal
        confidence = opportunity.signal_confidence
        
        # Adjust for risk-reward ratio
        rr_ratio = opportunity.risk_reward_ratio()
        if rr_ratio > 2.0:
            confidence *= 1.2
        elif rr_ratio < 1.0:
            confidence *= 0.8
        
        # Adjust for market conditions
        if market_condition.is_high_volatility():
            confidence *= 0.9
        
        # Adjust for liquidity
        if opportunity.liquidity_score < 0.5:
            confidence *= 0.8
        
        # Clamp to [0, 1]
        confidence = max(0.0, min(1.0, confidence))
        
        return confidence
    
    def get_sizing_summary(self) -> Dict[str, Any]:
        """Get comprehensive sizing summary"""
        
        if not self.sizing_history:
            return {"status": "No sizing history"}
        
        # Calculate performance metrics
        recent_results = self.sizing_history[-100:]
        
        avg_position_size = np.mean([r.position_size for r in recent_results])
        avg_risk_amount = np.mean([r.risk_amount for r in recent_results])
        avg_confidence = np.mean([r.confidence_score for r in recent_results])
        
        # Method usage
        method_usage = {}
        for result in recent_results:
            method = result.method_used
            method_usage[method] = method_usage.get(method, 0) + 1
        
        # Performance statistics
        avg_calc_time = np.mean(self.calculation_times) if self.calculation_times else 0
        
        return {
            "timestamp": datetime.now().isoformat(),
            "performance_metrics": {
                "avg_position_size": avg_position_size,
                "avg_risk_amount": avg_risk_amount,
                "avg_confidence": avg_confidence,
                "avg_calc_time_ms": avg_calc_time
            },
            "method_usage": method_usage,
            "sizing_history_count": len(self.sizing_history),
            "config": self.config.to_dict()
        }
    
    def optimize_portfolio_allocation(
        self,
        opportunities: List[TradeOpportunity],
        portfolio_state: PortfolioState,
        market_condition: MarketCondition
    ) -> Dict[str, float]:
        """
        Optimize portfolio allocation across multiple opportunities
        
        Args:
            opportunities: List of trade opportunities
            portfolio_state: Current portfolio state
            market_condition: Current market conditions
        
        Returns:
            Dictionary of symbol -> optimal allocation
        """
        
        if not opportunities:
            return {}
        
        # Prepare data for optimization
        expected_returns = np.array([opp.expected_return for opp in opportunities])
        volatilities = np.array([opp.expected_volatility for opp in opportunities])
        
        # Create covariance matrix (simplified)
        n_assets = len(opportunities)
        covariance_matrix = np.eye(n_assets)
        
        for i in range(n_assets):
            for j in range(n_assets):
                if i != j:
                    # Estimate correlation based on sectors
                    if opportunities[i].sector == opportunities[j].sector:
                        correlation = 0.6
                    else:
                        correlation = 0.3
                    
                    covariance_matrix[i, j] = correlation * volatilities[i] * volatilities[j]
                else:
                    covariance_matrix[i, j] = volatilities[i] ** 2
        
        # Set maximum weights
        max_weights = np.full(n_assets, self.config.max_position_size)
        
        # Optimize using JIT function
        optimal_weights = optimize_position_sizes(
            expected_returns,
            covariance_matrix,
            max_weights,
            self.config.max_portfolio_risk
        )
        
        # Create result dictionary
        result = {}
        for i, opp in enumerate(opportunities):
            if i < len(optimal_weights):
                result[opp.symbol] = optimal_weights[i]
        
        return result


# Factory function for creating position sizer
def create_position_sizer(config_dict: Optional[Dict[str, Any]] = None) -> AdvancedPositionSizer:
    """
    Create an advanced position sizer with configuration
    
    Args:
        config_dict: Configuration dictionary
    
    Returns:
        AdvancedPositionSizer instance
    """
    
    if config_dict is None:
        config = PositionSizingConfig()
    else:
        config = PositionSizingConfig(**config_dict)
    
    return AdvancedPositionSizer(config)


# Convenience functions for standalone usage
def calculate_kelly_position_size(
    win_probability: float,
    win_amount: float,
    loss_amount: float,
    safety_factor: float = 0.5,
    max_fraction: float = 0.25
) -> float:
    """Calculate Kelly position size"""
    
    kelly_fraction = calculate_kelly_fraction(
        win_probability, win_amount, loss_amount, safety_factor
    )
    
    return min(kelly_fraction, max_fraction)


def calculate_volatility_position_size(
    target_risk: float,
    asset_volatility: float,
    stop_distance: float
) -> float:
    """Calculate volatility-based position size"""
    
    if asset_volatility <= 0 or stop_distance <= 0:
        return 0.0
    
    position_risk = asset_volatility * stop_distance
    
    if position_risk <= 0:
        return 0.0
    
    return target_risk / position_risk