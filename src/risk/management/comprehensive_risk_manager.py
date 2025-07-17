"""
Comprehensive Risk Management Framework
======================================

This module implements a comprehensive risk management system that includes:
- Real-time risk monitoring and alerts
- Stop-loss and take-profit optimization
- Dynamic risk limits based on performance
- Portfolio-level risk controls
- Stress testing and scenario analysis
- Advanced risk measures integration

Author: Risk Management System
Date: 2025-07-17
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio
import logging
from numba import jit, njit
from scipy import stats
from enum import Enum
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk level enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(Enum):
    """Alert type enumeration"""
    VAR_BREACH = "var_breach"
    DRAWDOWN_WARNING = "drawdown_warning"
    LEVERAGE_WARNING = "leverage_warning"
    CORRELATION_SPIKE = "correlation_spike"
    POSITION_LIMIT = "position_limit"
    STOP_LOSS_HIT = "stop_loss_hit"
    TAKE_PROFIT_HIT = "take_profit_hit"
    MARGIN_CALL = "margin_call"
    SYSTEM_ERROR = "system_error"


class RiskAction(Enum):
    """Risk action enumeration"""
    MONITOR = "monitor"
    REDUCE_POSITION = "reduce_position"
    CLOSE_POSITION = "close_position"
    STOP_TRADING = "stop_trading"
    HEDGE_POSITION = "hedge_position"
    REBALANCE = "rebalance"


@dataclass
class RiskLimit:
    """Risk limit configuration"""
    name: str
    value: float
    warning_threshold: float
    critical_threshold: float
    enabled: bool = True
    action_on_breach: RiskAction = RiskAction.MONITOR
    
    def check_breach(self, current_value: float) -> Optional[RiskLevel]:
        """Check if limit is breached"""
        if not self.enabled:
            return None
        
        if current_value >= self.critical_threshold:
            return RiskLevel.CRITICAL
        elif current_value >= self.warning_threshold:
            return RiskLevel.HIGH
        elif current_value >= self.value * 0.8:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW


@dataclass
class RiskAlert:
    """Risk alert data"""
    timestamp: datetime
    alert_type: AlertType
    risk_level: RiskLevel
    message: str
    current_value: float
    threshold_value: float
    symbol: Optional[str] = None
    action_required: RiskAction = RiskAction.MONITOR
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'alert_type': self.alert_type.value,
            'risk_level': self.risk_level.value,
            'message': self.message,
            'current_value': self.current_value,
            'threshold_value': self.threshold_value,
            'symbol': self.symbol,
            'action_required': self.action_required.value,
            'metadata': self.metadata
        }


@dataclass
class PositionRisk:
    """Position risk data"""
    symbol: str
    position_size: float
    entry_price: float
    current_price: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    unrealized_pnl: float
    risk_amount: float
    time_in_trade: timedelta
    volatility: float
    beta: float
    correlation_risk: float
    liquidity_risk: float
    
    def get_risk_metrics(self) -> Dict[str, float]:
        """Get position risk metrics"""
        return {
            'position_size': self.position_size,
            'unrealized_pnl': self.unrealized_pnl,
            'risk_amount': self.risk_amount,
            'volatility': self.volatility,
            'beta': self.beta,
            'correlation_risk': self.correlation_risk,
            'liquidity_risk': self.liquidity_risk
        }


@dataclass
class PortfolioRisk:
    """Portfolio risk data"""
    total_value: float
    total_risk: float
    var_95: float
    var_99: float
    expected_shortfall: float
    max_drawdown: float
    current_drawdown: float
    leverage: float
    concentration_risk: float
    correlation_risk: float
    liquidity_risk: float
    positions: Dict[str, PositionRisk]
    
    def get_risk_summary(self) -> Dict[str, float]:
        """Get portfolio risk summary"""
        return {
            'total_value': self.total_value,
            'total_risk': self.total_risk,
            'var_95': self.var_95,
            'var_99': self.var_99,
            'expected_shortfall': self.expected_shortfall,
            'max_drawdown': self.max_drawdown,
            'current_drawdown': self.current_drawdown,
            'leverage': self.leverage,
            'concentration_risk': self.concentration_risk,
            'correlation_risk': self.correlation_risk,
            'liquidity_risk': self.liquidity_risk
        }


@dataclass
class RiskManagementConfig:
    """Risk management configuration"""
    # Portfolio limits
    max_portfolio_risk: float = 0.15
    max_position_size: float = 0.10
    max_correlation: float = 0.7
    max_leverage: float = 3.0
    max_drawdown: float = 0.20
    daily_loss_limit: float = 0.05
    
    # VaR parameters
    var_confidence_levels: List[float] = field(default_factory=lambda: [0.95, 0.99])
    var_time_horizons: List[int] = field(default_factory=lambda: [1, 5, 10])
    var_lookback_period: int = 252
    
    # Stop loss parameters
    default_stop_loss: float = 0.02
    max_stop_loss: float = 0.05
    min_stop_loss: float = 0.01
    trailing_stop_enabled: bool = True
    
    # Take profit parameters
    default_take_profit: float = 0.04
    min_risk_reward_ratio: float = 1.5
    partial_profit_enabled: bool = True
    
    # Monitoring parameters
    monitoring_frequency: int = 5  # seconds
    alert_cooldown: int = 300  # seconds
    
    # Circuit breaker levels
    circuit_breaker_1: float = 0.10  # 10% drawdown
    circuit_breaker_2: float = 0.15  # 15% drawdown  
    circuit_breaker_3: float = 0.20  # 20% drawdown
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'max_portfolio_risk': self.max_portfolio_risk,
            'max_position_size': self.max_position_size,
            'max_correlation': self.max_correlation,
            'max_leverage': self.max_leverage,
            'max_drawdown': self.max_drawdown,
            'daily_loss_limit': self.daily_loss_limit,
            'var_confidence_levels': self.var_confidence_levels,
            'var_time_horizons': self.var_time_horizons,
            'var_lookback_period': self.var_lookback_period,
            'default_stop_loss': self.default_stop_loss,
            'max_stop_loss': self.max_stop_loss,
            'min_stop_loss': self.min_stop_loss,
            'trailing_stop_enabled': self.trailing_stop_enabled,
            'default_take_profit': self.default_take_profit,
            'min_risk_reward_ratio': self.min_risk_reward_ratio,
            'partial_profit_enabled': self.partial_profit_enabled,
            'monitoring_frequency': self.monitoring_frequency,
            'alert_cooldown': self.alert_cooldown,
            'circuit_breaker_1': self.circuit_breaker_1,
            'circuit_breaker_2': self.circuit_breaker_2,
            'circuit_breaker_3': self.circuit_breaker_3
        }


# JIT optimized risk calculation functions
@njit
def calculate_portfolio_var(
    position_values: np.ndarray,
    correlation_matrix: np.ndarray,
    volatilities: np.ndarray,
    confidence_level: float = 0.95
) -> float:
    """Calculate portfolio VaR - JIT optimized"""
    
    if len(position_values) == 0:
        return 0.0
    
    # Portfolio weights
    total_value = np.sum(position_values)
    if total_value == 0:
        return 0.0
    
    weights = position_values / total_value
    
    # Portfolio variance
    portfolio_variance = np.dot(weights, np.dot(correlation_matrix * np.outer(volatilities, volatilities), weights))
    
    # Portfolio volatility
    portfolio_volatility = np.sqrt(max(0.0, portfolio_variance))
    
    # VaR calculation (assuming normal distribution)
    z_score = stats.norm.ppf(1 - confidence_level)
    var = portfolio_volatility * z_score * total_value
    
    return abs(var)


@njit
def calculate_marginal_var(
    position_values: np.ndarray,
    correlation_matrix: np.ndarray,
    volatilities: np.ndarray,
    asset_index: int,
    confidence_level: float = 0.95
) -> float:
    """Calculate marginal VaR for an asset - JIT optimized"""
    
    if len(position_values) == 0 or asset_index >= len(position_values):
        return 0.0
    
    # Portfolio weights
    total_value = np.sum(position_values)
    if total_value == 0:
        return 0.0
    
    weights = position_values / total_value
    
    # Covariance matrix
    covariance_matrix = correlation_matrix * np.outer(volatilities, volatilities)
    
    # Portfolio variance
    portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
    
    if portfolio_variance <= 0:
        return 0.0
    
    # Marginal contribution to portfolio variance
    marginal_contribution = np.dot(covariance_matrix[asset_index], weights) / np.sqrt(portfolio_variance)
    
    # Convert to VaR
    z_score = stats.norm.ppf(1 - confidence_level)
    marginal_var = marginal_contribution * z_score * total_value
    
    return abs(marginal_var)


@njit
def calculate_expected_shortfall(
    portfolio_returns: np.ndarray,
    confidence_level: float = 0.95
) -> float:
    """Calculate Expected Shortfall - JIT optimized"""
    
    if len(portfolio_returns) == 0:
        return 0.0
    
    # Sort returns
    sorted_returns = np.sort(portfolio_returns)
    
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
def calculate_concentration_risk(position_weights: np.ndarray) -> float:
    """Calculate concentration risk using Herfindahl index - JIT optimized"""
    
    if len(position_weights) == 0:
        return 0.0
    
    # Herfindahl index
    herfindahl_index = np.sum(position_weights ** 2)
    
    # Normalize to [0, 1] range
    n = len(position_weights)
    if n <= 1:
        return 1.0
    
    normalized_concentration = (herfindahl_index - 1/n) / (1 - 1/n)
    
    return max(0.0, normalized_concentration)


@njit
def calculate_correlation_risk(
    correlation_matrix: np.ndarray,
    position_weights: np.ndarray
) -> float:
    """Calculate correlation risk - JIT optimized"""
    
    if len(correlation_matrix) == 0 or len(position_weights) == 0:
        return 0.0
    
    # Weighted average correlation
    weighted_correlation = 0.0
    total_weight = 0.0
    
    for i in range(len(position_weights)):
        for j in range(len(position_weights)):
            if i != j:
                weight = position_weights[i] * position_weights[j]
                weighted_correlation += weight * correlation_matrix[i, j]
                total_weight += weight
    
    if total_weight == 0:
        return 0.0
    
    return weighted_correlation / total_weight


class ComprehensiveRiskManager:
    """
    Comprehensive Risk Management System
    
    This class implements a complete risk management framework with:
    - Real-time risk monitoring
    - Dynamic risk limits
    - Stop-loss and take-profit optimization
    - Portfolio-level risk controls
    - Alert system
    """
    
    def __init__(self, config: RiskManagementConfig):
        """
        Initialize the comprehensive risk manager
        
        Args:
            config: Risk management configuration
        """
        self.config = config
        
        # Risk limits
        self.risk_limits = self._initialize_risk_limits()
        
        # Alert system
        self.alerts: List[RiskAlert] = []
        self.alert_handlers: Dict[AlertType, List[Callable]] = {}
        self.last_alert_time: Dict[str, datetime] = {}
        
        # Portfolio state
        self.portfolio_risk: Optional[PortfolioRisk] = None
        self.position_risks: Dict[str, PositionRisk] = {}
        
        # Performance tracking
        self.risk_history: List[Dict[str, float]] = []
        self.calculation_times: List[float] = []
        
        # Circuit breaker state
        self.circuit_breaker_active = False
        self.circuit_breaker_level = 0
        
        # Monitoring
        self.monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None
        
        logger.info("ComprehensiveRiskManager initialized",
                   extra={'config': config.to_dict()})
    
    def _initialize_risk_limits(self) -> Dict[str, RiskLimit]:
        """Initialize risk limits"""
        return {
            'portfolio_risk': RiskLimit(
                name='portfolio_risk',
                value=self.config.max_portfolio_risk,
                warning_threshold=self.config.max_portfolio_risk * 0.8,
                critical_threshold=self.config.max_portfolio_risk,
                action_on_breach=RiskAction.REDUCE_POSITION
            ),
            'position_size': RiskLimit(
                name='position_size',
                value=self.config.max_position_size,
                warning_threshold=self.config.max_position_size * 0.8,
                critical_threshold=self.config.max_position_size,
                action_on_breach=RiskAction.REDUCE_POSITION
            ),
            'leverage': RiskLimit(
                name='leverage',
                value=self.config.max_leverage,
                warning_threshold=self.config.max_leverage * 0.8,
                critical_threshold=self.config.max_leverage,
                action_on_breach=RiskAction.REDUCE_POSITION
            ),
            'drawdown': RiskLimit(
                name='drawdown',
                value=self.config.max_drawdown,
                warning_threshold=self.config.max_drawdown * 0.8,
                critical_threshold=self.config.max_drawdown,
                action_on_breach=RiskAction.STOP_TRADING
            ),
            'daily_loss': RiskLimit(
                name='daily_loss',
                value=self.config.daily_loss_limit,
                warning_threshold=self.config.daily_loss_limit * 0.8,
                critical_threshold=self.config.daily_loss_limit,
                action_on_breach=RiskAction.STOP_TRADING
            )
        }
    
    async def update_portfolio_risk(
        self,
        positions: Dict[str, Any],
        market_data: Dict[str, Any],
        portfolio_value: float
    ) -> PortfolioRisk:
        """
        Update portfolio risk assessment
        
        Args:
            positions: Current positions
            market_data: Market data
            portfolio_value: Total portfolio value
        
        Returns:
            Updated portfolio risk
        """
        start_time = datetime.now()
        
        try:
            # Update position risks
            position_risks = {}
            for symbol, position in positions.items():
                position_risk = await self._calculate_position_risk(
                    symbol, position, market_data.get(symbol, {})
                )
                position_risks[symbol] = position_risk
            
            # Calculate portfolio-level metrics
            portfolio_risk = await self._calculate_portfolio_risk(
                position_risks, portfolio_value
            )
            
            # Update state
            self.portfolio_risk = portfolio_risk
            self.position_risks = position_risks
            
            # Check risk limits
            await self._check_risk_limits(portfolio_risk)
            
            # Store in history
            self.risk_history.append({
                'timestamp': datetime.now().isoformat(),
                'total_risk': portfolio_risk.total_risk,
                'var_95': portfolio_risk.var_95,
                'current_drawdown': portfolio_risk.current_drawdown,
                'leverage': portfolio_risk.leverage
            })
            
            # Keep only recent history
            if len(self.risk_history) > 10000:
                self.risk_history = self.risk_history[-10000:]
            
            return portfolio_risk
            
        except Exception as e:
            logger.error(f"Portfolio risk update failed: {e}")
            raise
        
        finally:
            # Track performance
            calc_time = (datetime.now() - start_time).total_seconds() * 1000
            self.calculation_times.append(calc_time)
            
            if len(self.calculation_times) > 1000:
                self.calculation_times = self.calculation_times[-1000:]
    
    async def _calculate_position_risk(
        self,
        symbol: str,
        position: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> PositionRisk:
        """Calculate risk for individual position"""
        
        # Extract position data
        position_size = position.get('size', 0.0)
        entry_price = position.get('entry_price', 0.0)
        current_price = market_data.get('price', entry_price)
        
        # Calculate P&L
        if position_size != 0:
            unrealized_pnl = (current_price - entry_price) * position_size
        else:
            unrealized_pnl = 0.0
        
        # Risk metrics
        volatility = market_data.get('volatility', 0.2)
        beta = market_data.get('beta', 1.0)
        
        # Calculate risk amount
        risk_amount = abs(position_size) * volatility * current_price
        
        # Time in trade
        entry_time = position.get('entry_time', datetime.now())
        if isinstance(entry_time, str):
            entry_time = datetime.fromisoformat(entry_time)
        time_in_trade = datetime.now() - entry_time
        
        return PositionRisk(
            symbol=symbol,
            position_size=position_size,
            entry_price=entry_price,
            current_price=current_price,
            stop_loss=position.get('stop_loss'),
            take_profit=position.get('take_profit'),
            unrealized_pnl=unrealized_pnl,
            risk_amount=risk_amount,
            time_in_trade=time_in_trade,
            volatility=volatility,
            beta=beta,
            correlation_risk=0.0,  # Will be calculated at portfolio level
            liquidity_risk=market_data.get('liquidity_risk', 0.0)
        )
    
    async def _calculate_portfolio_risk(
        self,
        position_risks: Dict[str, PositionRisk],
        portfolio_value: float
    ) -> PortfolioRisk:
        """Calculate portfolio-level risk metrics"""
        
        if not position_risks:
            return PortfolioRisk(
                total_value=portfolio_value,
                total_risk=0.0,
                var_95=0.0,
                var_99=0.0,
                expected_shortfall=0.0,
                max_drawdown=0.0,
                current_drawdown=0.0,
                leverage=1.0,
                concentration_risk=0.0,
                correlation_risk=0.0,
                liquidity_risk=0.0,
                positions=position_risks
            )
        
        # Extract data for calculations
        symbols = list(position_risks.keys())
        position_values = np.array([abs(pos.position_size * pos.current_price) for pos in position_risks.values()])
        volatilities = np.array([pos.volatility for pos in position_risks.values()])
        
        # Create correlation matrix (simplified)
        n_assets = len(symbols)
        correlation_matrix = np.eye(n_assets)
        for i in range(n_assets):
            for j in range(n_assets):
                if i != j:
                    correlation_matrix[i, j] = 0.3  # Assume moderate correlation
        
        # Calculate VaR
        var_95 = calculate_portfolio_var(position_values, correlation_matrix, volatilities, 0.95)
        var_99 = calculate_portfolio_var(position_values, correlation_matrix, volatilities, 0.99)
        
        # Calculate Expected Shortfall (simplified)
        portfolio_returns = np.array([pos.unrealized_pnl / portfolio_value for pos in position_risks.values()])
        expected_shortfall = calculate_expected_shortfall(portfolio_returns, 0.95)
        
        # Calculate concentration risk
        position_weights = position_values / np.sum(position_values) if np.sum(position_values) > 0 else np.zeros_like(position_values)
        concentration_risk = calculate_concentration_risk(position_weights)
        
        # Calculate correlation risk
        correlation_risk = calculate_correlation_risk(correlation_matrix, position_weights)
        
        # Calculate other metrics
        total_risk = np.sum([pos.risk_amount for pos in position_risks.values()])
        leverage = np.sum(position_values) / portfolio_value if portfolio_value > 0 else 1.0
        
        # Calculate drawdown (simplified)
        current_pnl = sum(pos.unrealized_pnl for pos in position_risks.values())
        current_drawdown = max(0.0, -current_pnl / portfolio_value) if portfolio_value > 0 else 0.0
        
        return PortfolioRisk(
            total_value=portfolio_value,
            total_risk=total_risk,
            var_95=var_95,
            var_99=var_99,
            expected_shortfall=expected_shortfall,
            max_drawdown=0.0,  # This would need historical data
            current_drawdown=current_drawdown,
            leverage=leverage,
            concentration_risk=concentration_risk,
            correlation_risk=correlation_risk,
            liquidity_risk=0.0,  # Simplified
            positions=position_risks
        )
    
    async def _check_risk_limits(self, portfolio_risk: PortfolioRisk) -> None:
        """Check risk limits and generate alerts"""
        
        # Check portfolio risk limit
        portfolio_risk_level = self.risk_limits['portfolio_risk'].check_breach(
            portfolio_risk.total_risk / portfolio_risk.total_value if portfolio_risk.total_value > 0 else 0
        )
        
        if portfolio_risk_level and portfolio_risk_level.value in ['high', 'critical']:
            await self._generate_alert(
                AlertType.VAR_BREACH,
                portfolio_risk_level,
                f"Portfolio risk exceeded threshold: {portfolio_risk.total_risk:.2%}",
                portfolio_risk.total_risk,
                self.config.max_portfolio_risk
            )
        
        # Check leverage limit
        leverage_level = self.risk_limits['leverage'].check_breach(portfolio_risk.leverage)
        
        if leverage_level and leverage_level.value in ['high', 'critical']:
            await self._generate_alert(
                AlertType.LEVERAGE_WARNING,
                leverage_level,
                f"Leverage exceeded threshold: {portfolio_risk.leverage:.2f}x",
                portfolio_risk.leverage,
                self.config.max_leverage
            )
        
        # Check drawdown limit
        drawdown_level = self.risk_limits['drawdown'].check_breach(portfolio_risk.current_drawdown)
        
        if drawdown_level and drawdown_level.value in ['high', 'critical']:
            await self._generate_alert(
                AlertType.DRAWDOWN_WARNING,
                drawdown_level,
                f"Drawdown exceeded threshold: {portfolio_risk.current_drawdown:.2%}",
                portfolio_risk.current_drawdown,
                self.config.max_drawdown
            )
            
            # Check circuit breakers
            await self._check_circuit_breakers(portfolio_risk.current_drawdown)
    
    async def _generate_alert(
        self,
        alert_type: AlertType,
        risk_level: RiskLevel,
        message: str,
        current_value: float,
        threshold_value: float,
        symbol: Optional[str] = None
    ) -> None:
        """Generate risk alert"""
        
        # Check cooldown
        alert_key = f"{alert_type.value}_{symbol or 'portfolio'}"
        last_alert = self.last_alert_time.get(alert_key)
        
        if last_alert and (datetime.now() - last_alert).seconds < self.config.alert_cooldown:
            return
        
        # Create alert
        alert = RiskAlert(
            timestamp=datetime.now(),
            alert_type=alert_type,
            risk_level=risk_level,
            message=message,
            current_value=current_value,
            threshold_value=threshold_value,
            symbol=symbol,
            action_required=self._determine_action(alert_type, risk_level)
        )
        
        # Store alert
        self.alerts.append(alert)
        self.last_alert_time[alert_key] = datetime.now()
        
        # Keep only recent alerts
        if len(self.alerts) > 10000:
            self.alerts = self.alerts[-10000:]
        
        # Execute handlers
        handlers = self.alert_handlers.get(alert_type, [])
        for handler in handlers:
            try:
                await handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")
        
        logger.warning(f"Risk alert generated: {alert.message}",
                      extra={'alert': alert.to_dict()})
    
    def _determine_action(self, alert_type: AlertType, risk_level: RiskLevel) -> RiskAction:
        """Determine required action based on alert"""
        
        if risk_level == RiskLevel.CRITICAL:
            if alert_type in [AlertType.DRAWDOWN_WARNING, AlertType.MARGIN_CALL]:
                return RiskAction.STOP_TRADING
            else:
                return RiskAction.REDUCE_POSITION
        elif risk_level == RiskLevel.HIGH:
            return RiskAction.REDUCE_POSITION
        else:
            return RiskAction.MONITOR
    
    async def _check_circuit_breakers(self, current_drawdown: float) -> None:
        """Check circuit breaker levels"""
        
        if current_drawdown >= self.config.circuit_breaker_3:
            await self._activate_circuit_breaker(3)
        elif current_drawdown >= self.config.circuit_breaker_2:
            await self._activate_circuit_breaker(2)
        elif current_drawdown >= self.config.circuit_breaker_1:
            await self._activate_circuit_breaker(1)
    
    async def _activate_circuit_breaker(self, level: int) -> None:
        """Activate circuit breaker"""
        
        if self.circuit_breaker_level >= level:
            return  # Already activated
        
        self.circuit_breaker_active = True
        self.circuit_breaker_level = level
        
        await self._generate_alert(
            AlertType.DRAWDOWN_WARNING,
            RiskLevel.CRITICAL,
            f"Circuit breaker level {level} activated",
            0.0,
            0.0
        )
        
        logger.critical(f"Circuit breaker level {level} activated")
    
    async def optimize_stop_loss(
        self,
        symbol: str,
        position_size: float,
        entry_price: float,
        current_price: float,
        volatility: float,
        market_data: Dict[str, Any]
    ) -> float:
        """
        Optimize stop-loss level
        
        Args:
            symbol: Asset symbol
            position_size: Position size
            entry_price: Entry price
            current_price: Current price
            volatility: Asset volatility
            market_data: Additional market data
        
        Returns:
            Optimized stop-loss level
        """
        
        # Base stop loss
        base_stop = self.config.default_stop_loss
        
        # Volatility adjustment
        vol_adjustment = volatility / 0.2  # Normalize to 20% volatility
        adjusted_stop = base_stop * vol_adjustment
        
        # Apply limits
        adjusted_stop = max(self.config.min_stop_loss, adjusted_stop)
        adjusted_stop = min(self.config.max_stop_loss, adjusted_stop)
        
        # Calculate stop price
        if position_size > 0:  # Long position
            stop_price = entry_price * (1 - adjusted_stop)
        else:  # Short position
            stop_price = entry_price * (1 + adjusted_stop)
        
        return stop_price
    
    async def optimize_take_profit(
        self,
        symbol: str,
        position_size: float,
        entry_price: float,
        stop_loss: float,
        risk_reward_ratio: float = None
    ) -> float:
        """
        Optimize take-profit level
        
        Args:
            symbol: Asset symbol
            position_size: Position size
            entry_price: Entry price
            stop_loss: Stop-loss level
            risk_reward_ratio: Desired risk-reward ratio
        
        Returns:
            Optimized take-profit level
        """
        
        if risk_reward_ratio is None:
            risk_reward_ratio = self.config.min_risk_reward_ratio
        
        # Calculate risk amount
        risk_amount = abs(entry_price - stop_loss)
        
        # Calculate reward amount
        reward_amount = risk_amount * risk_reward_ratio
        
        # Calculate take profit price
        if position_size > 0:  # Long position
            take_profit_price = entry_price + reward_amount
        else:  # Short position
            take_profit_price = entry_price - reward_amount
        
        return take_profit_price
    
    async def start_monitoring(self) -> None:
        """Start real-time risk monitoring"""
        
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Risk monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop real-time risk monitoring"""
        
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Risk monitoring stopped")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        
        while self.monitoring_active:
            try:
                # This would be called by external system with real data
                # For now, just sleep
                await asyncio.sleep(self.config.monitoring_frequency)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(self.config.monitoring_frequency)
    
    def add_alert_handler(self, alert_type: AlertType, handler: Callable) -> None:
        """Add alert handler"""
        
        if alert_type not in self.alert_handlers:
            self.alert_handlers[alert_type] = []
        
        self.alert_handlers[alert_type].append(handler)
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary"""
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'portfolio_risk': self.portfolio_risk.get_risk_summary() if self.portfolio_risk else {},
            'circuit_breaker_active': self.circuit_breaker_active,
            'circuit_breaker_level': self.circuit_breaker_level,
            'monitoring_active': self.monitoring_active,
            'recent_alerts': [alert.to_dict() for alert in self.alerts[-10:]],
            'risk_limits': {name: {
                'value': limit.value,
                'warning_threshold': limit.warning_threshold,
                'critical_threshold': limit.critical_threshold,
                'enabled': limit.enabled
            } for name, limit in self.risk_limits.items()},
            'performance': {
                'avg_calc_time_ms': np.mean(self.calculation_times) if self.calculation_times else 0,
                'risk_history_count': len(self.risk_history),
                'alert_count': len(self.alerts)
            }
        }
        
        return summary
    
    def get_position_risk_summary(self, symbol: str) -> Dict[str, Any]:
        """Get risk summary for specific position"""
        
        position_risk = self.position_risks.get(symbol)
        
        if not position_risk:
            return {'symbol': symbol, 'status': 'No position'}
        
        return {
            'symbol': symbol,
            'risk_metrics': position_risk.get_risk_metrics(),
            'stop_loss': position_risk.stop_loss,
            'take_profit': position_risk.take_profit,
            'time_in_trade': position_risk.time_in_trade.total_seconds() / 3600,  # hours
            'unrealized_pnl': position_risk.unrealized_pnl
        }


# Factory function
def create_risk_manager(config_dict: Optional[Dict[str, Any]] = None) -> ComprehensiveRiskManager:
    """
    Create a comprehensive risk manager with configuration
    
    Args:
        config_dict: Configuration dictionary
    
    Returns:
        ComprehensiveRiskManager instance
    """
    
    if config_dict is None:
        config = RiskManagementConfig()
    else:
        config = RiskManagementConfig(**config_dict)
    
    return ComprehensiveRiskManager(config)