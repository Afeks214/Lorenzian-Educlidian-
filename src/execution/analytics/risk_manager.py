"""
Pre-Trade Risk Manager

Comprehensive pre-trade risk validation and monitoring for order execution.
Integrates with the broader risk management system for portfolio-level controls.
"""

import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import structlog

from ..order_management.order_types import Order

logger = structlog.get_logger()


class RiskLevel(Enum):
    """Risk level classifications"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class RiskCheckResult:
    """Result of pre-trade risk check"""
    
    approved: bool
    risk_level: RiskLevel
    risk_score: float  # 0-1 scale
    
    # Failed checks
    failed_checks: List[str]
    warnings: List[str]
    
    # Risk factors
    risk_factors: Dict[str, float]
    
    # Recommendations
    recommended_adjustments: Dict[str, Any]
    
    # Timing
    check_duration_ms: float
    timestamp: datetime
    
    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0
    
    @property
    def needs_approval(self) -> bool:
        return self.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]


@dataclass
class RiskLimits:
    """Risk limits configuration"""
    
    # Position limits
    max_position_size: Dict[str, int]  # symbol -> max position
    max_position_value: Dict[str, float]  # symbol -> max value
    max_sector_exposure: Dict[str, float]  # sector -> max exposure
    
    # Order limits
    max_order_size: int
    max_order_value: float
    max_daily_volume: int
    max_orders_per_minute: int
    
    # Concentration limits
    max_single_name_concentration: float
    max_sector_concentration: float
    max_venue_concentration: float
    
    # Leverage limits
    max_gross_leverage: float
    max_net_leverage: float
    
    # Market risk limits
    max_portfolio_var: float
    max_symbol_var: float
    stress_test_threshold: float


class PreTradeRiskManager:
    """
    Pre-trade risk validation and monitoring.
    
    Performs comprehensive risk checks before order execution including:
    - Position limit validation
    - Concentration risk assessment
    - Market risk evaluation
    - Compliance checks
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize risk limits
        self.risk_limits = self._initialize_risk_limits(config)
        
        # Risk monitoring state
        self.daily_volumes: Dict[str, int] = {}  # symbol -> volume
        self.order_counts: Dict[str, int] = {}   # time window -> count
        self.position_cache: Dict[str, int] = {} # symbol -> position
        
        # Performance tracking
        self.check_latencies: List[float] = []
        self.approval_rates: Dict[RiskLevel, float] = {}
        
        logger.info("PreTradeRiskManager initialized")
    
    def _initialize_risk_limits(self, config: Dict[str, Any]) -> RiskLimits:
        """Initialize risk limits from configuration"""
        
        return RiskLimits(
            max_position_size=config.get('max_position_size', {}),
            max_position_value=config.get('max_position_value', {}),
            max_sector_exposure=config.get('max_sector_exposure', {}),
            
            max_order_size=config.get('max_order_size', 100000),
            max_order_value=config.get('max_order_value', 10000000),
            max_daily_volume=config.get('max_daily_volume', 1000000),
            max_orders_per_minute=config.get('max_orders_per_minute', 100),
            
            max_single_name_concentration=config.get('max_single_name_concentration', 0.10),
            max_sector_concentration=config.get('max_sector_concentration', 0.25),
            max_venue_concentration=config.get('max_venue_concentration', 0.50),
            
            max_gross_leverage=config.get('max_gross_leverage', 2.0),
            max_net_leverage=config.get('max_net_leverage', 1.0),
            
            max_portfolio_var=config.get('max_portfolio_var', 0.02),
            max_symbol_var=config.get('max_symbol_var', 0.05),
            stress_test_threshold=config.get('stress_test_threshold', 0.10)
        )
    
    async def check_order_risk(
        self,
        order: Order,
        portfolio_state: Optional[Dict[str, Any]] = None
    ) -> RiskCheckResult:
        """
        Perform comprehensive pre-trade risk check.
        Target: <10ms execution time for real-time validation.
        """
        start_time = time.perf_counter()
        
        try:
            failed_checks = []
            warnings = []
            risk_factors = {}
            
            # Basic order validation
            basic_result = self._check_basic_order_limits(order)
            failed_checks.extend(basic_result['failed'])
            warnings.extend(basic_result['warnings'])
            risk_factors.update(basic_result['factors'])
            
            # Position limit checks
            position_result = await self._check_position_limits(order, portfolio_state)
            failed_checks.extend(position_result['failed'])
            warnings.extend(position_result['warnings'])
            risk_factors.update(position_result['factors'])
            
            # Concentration checks
            concentration_result = await self._check_concentration_limits(order, portfolio_state)
            failed_checks.extend(concentration_result['failed'])
            warnings.extend(concentration_result['warnings'])
            risk_factors.update(concentration_result['factors'])
            
            # Market risk checks
            market_risk_result = await self._check_market_risk(order, portfolio_state)
            failed_checks.extend(market_risk_result['failed'])
            warnings.extend(market_risk_result['warnings'])
            risk_factors.update(market_risk_result['factors'])
            
            # Volume and frequency checks
            volume_result = self._check_volume_limits(order)
            failed_checks.extend(volume_result['failed'])
            warnings.extend(volume_result['warnings'])
            risk_factors.update(volume_result['factors'])
            
            # Calculate overall risk score and level
            risk_score = self._calculate_risk_score(risk_factors)
            risk_level = self._determine_risk_level(risk_score, failed_checks)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(order, failed_checks, risk_factors)
            
            # Check duration
            check_duration = (time.perf_counter() - start_time) * 1000
            self.check_latencies.append(check_duration)
            
            # Keep only recent latencies
            if len(self.check_latencies) > 1000:
                self.check_latencies = self.check_latencies[-1000:]
            
            result = RiskCheckResult(
                approved=len(failed_checks) == 0,
                risk_level=risk_level,
                risk_score=risk_score,
                failed_checks=failed_checks,
                warnings=warnings,
                risk_factors=risk_factors,
                recommended_adjustments=recommendations,
                check_duration_ms=check_duration,
                timestamp=datetime.now()
            )
            
            # Update approval rates
            self._update_approval_rates(result)
            
            # Log high-risk orders
            if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                logger.warning(
                    "High-risk order detected",
                    order_id=order.order_id,
                    symbol=order.symbol,
                    risk_level=risk_level.value,
                    risk_score=risk_score,
                    failed_checks=failed_checks
                )
            
            return result
            
        except Exception as e:
            check_duration = (time.perf_counter() - start_time) * 1000
            logger.error(f"Risk check failed: {str(e)}")
            
            # Return conservative result on error
            return RiskCheckResult(
                approved=False,
                risk_level=RiskLevel.CRITICAL,
                risk_score=1.0,
                failed_checks=[f"Risk check error: {str(e)}"],
                warnings=[],
                risk_factors={'error': 1.0},
                recommended_adjustments={},
                check_duration_ms=check_duration,
                timestamp=datetime.now()
            )
    
    def _check_basic_order_limits(self, order: Order) -> Dict[str, List]:
        """Check basic order size and value limits"""
        
        failed = []
        warnings = []
        factors = {}
        
        # Order size check
        if order.quantity > self.risk_limits.max_order_size:
            failed.append(f"Order size {order.quantity} exceeds limit {self.risk_limits.max_order_size}")
            factors['order_size_violation'] = min(1.0, order.quantity / self.risk_limits.max_order_size)
        elif order.quantity > self.risk_limits.max_order_size * 0.8:
            warnings.append(f"Large order size: {order.quantity}")
            factors['large_order_size'] = order.quantity / self.risk_limits.max_order_size
        
        # Order value check
        order_value = order.notional_value
        if order_value > self.risk_limits.max_order_value:
            failed.append(f"Order value ${order_value:,.0f} exceeds limit ${self.risk_limits.max_order_value:,.0f}")
            factors['order_value_violation'] = min(1.0, order_value / self.risk_limits.max_order_value)
        elif order_value > self.risk_limits.max_order_value * 0.8:
            warnings.append(f"Large order value: ${order_value:,.0f}")
            factors['large_order_value'] = order_value / self.risk_limits.max_order_value
        
        return {'failed': failed, 'warnings': warnings, 'factors': factors}
    
    async def _check_position_limits(
        self,
        order: Order,
        portfolio_state: Optional[Dict[str, Any]]
    ) -> Dict[str, List]:
        """Check position limits and constraints"""
        
        failed = []
        warnings = []
        factors = {}
        
        # Get current position (from portfolio state or cache)
        current_position = 0
        if portfolio_state:
            positions = portfolio_state.get('positions', {})
            current_position = positions.get(order.symbol, 0)
        else:
            current_position = self.position_cache.get(order.symbol, 0)
        
        # Calculate new position after order
        position_change = order.signed_quantity
        new_position = current_position + position_change
        
        # Check symbol-specific position limit
        symbol_limit = self.risk_limits.max_position_size.get(
            order.symbol,
            self.risk_limits.max_position_size.get('default', 1000000)
        )
        
        if abs(new_position) > symbol_limit:
            failed.append(f"Position limit exceeded for {order.symbol}: {new_position} > {symbol_limit}")
            factors['position_limit_violation'] = abs(new_position) / symbol_limit
        elif abs(new_position) > symbol_limit * 0.9:
            warnings.append(f"Approaching position limit for {order.symbol}")
            factors['position_limit_warning'] = abs(new_position) / symbol_limit
        
        # Check position value limit
        estimated_price = order.price or self._get_estimated_price(order.symbol)
        new_position_value = abs(new_position * estimated_price)
        
        value_limit = self.risk_limits.max_position_value.get(
            order.symbol,
            self.risk_limits.max_position_value.get('default', 50000000)
        )
        
        if new_position_value > value_limit:
            failed.append(f"Position value limit exceeded for {order.symbol}")
            factors['position_value_violation'] = new_position_value / value_limit
        
        return {'failed': failed, 'warnings': warnings, 'factors': factors}
    
    async def _check_concentration_limits(
        self,
        order: Order,
        portfolio_state: Optional[Dict[str, Any]]
    ) -> Dict[str, List]:
        """Check concentration risk limits"""
        
        failed = []
        warnings = []
        factors = {}
        
        if not portfolio_state:
            # Can't check concentration without portfolio state
            return {'failed': failed, 'warnings': warnings, 'factors': factors}
        
        # Get portfolio total value
        total_portfolio_value = portfolio_state.get('total_value', 100000000)  # $100M default
        
        # Calculate order value
        order_value = order.notional_value
        
        # Single name concentration
        current_symbol_value = portfolio_state.get('positions', {}).get(order.symbol, 0) * \
                              self._get_estimated_price(order.symbol)
        new_symbol_value = current_symbol_value + order_value
        
        symbol_concentration = new_symbol_value / total_portfolio_value
        
        if symbol_concentration > self.risk_limits.max_single_name_concentration:
            failed.append(f"Single name concentration limit exceeded: {symbol_concentration:.1%}")
            factors['concentration_violation'] = symbol_concentration / self.risk_limits.max_single_name_concentration
        elif symbol_concentration > self.risk_limits.max_single_name_concentration * 0.9:
            warnings.append(f"High single name concentration: {symbol_concentration:.1%}")
            factors['concentration_warning'] = symbol_concentration / self.risk_limits.max_single_name_concentration
        
        # Sector concentration (simplified - would need sector mapping in production)
        sector = self._get_sector(order.symbol)
        if sector:
            sector_limit = self.risk_limits.max_sector_exposure.get(sector, self.risk_limits.max_sector_concentration)
            current_sector_exposure = portfolio_state.get('sector_exposures', {}).get(sector, 0)
            new_sector_exposure = (current_sector_exposure + order_value) / total_portfolio_value
            
            if new_sector_exposure > sector_limit:
                failed.append(f"Sector concentration limit exceeded for {sector}: {new_sector_exposure:.1%}")
                factors['sector_concentration_violation'] = new_sector_exposure / sector_limit
        
        return {'failed': failed, 'warnings': warnings, 'factors': factors}
    
    async def _check_market_risk(
        self,
        order: Order,
        portfolio_state: Optional[Dict[str, Any]]
    ) -> Dict[str, List]:
        """Check market risk limits"""
        
        failed = []
        warnings = []
        factors = {}
        
        # Simplified market risk checks
        # In production, would integrate with proper risk models
        
        # Volatility check
        symbol_volatility = self._get_symbol_volatility(order.symbol)
        if symbol_volatility > 0.05:  # 5% daily volatility
            warnings.append(f"High volatility symbol: {order.symbol} ({symbol_volatility:.1%})")
            factors['high_volatility'] = min(1.0, symbol_volatility / 0.10)
        
        # VaR impact estimate
        order_var_impact = self._estimate_var_impact(order)
        if portfolio_state:
            current_var = portfolio_state.get('portfolio_var', 0.01)
            new_estimated_var = current_var + order_var_impact
            
            if new_estimated_var > self.risk_limits.max_portfolio_var:
                failed.append(f"Portfolio VaR limit would be exceeded")
                factors['var_violation'] = new_estimated_var / self.risk_limits.max_portfolio_var
        
        # Stress test check
        stress_loss = self._estimate_stress_loss(order)
        if stress_loss > self.risk_limits.stress_test_threshold:
            warnings.append(f"High stress test impact: {stress_loss:.1%}")
            factors['stress_test_warning'] = stress_loss / self.risk_limits.stress_test_threshold
        
        return {'failed': failed, 'warnings': warnings, 'factors': factors}
    
    def _check_volume_limits(self, order: Order) -> Dict[str, List]:
        """Check volume and frequency limits"""
        
        failed = []
        warnings = []
        factors = {}
        
        # Daily volume check
        today = datetime.now().date().isoformat()
        current_daily_volume = self.daily_volumes.get(f"{order.symbol}_{today}", 0)
        new_daily_volume = current_daily_volume + order.quantity
        
        if new_daily_volume > self.risk_limits.max_daily_volume:
            failed.append(f"Daily volume limit exceeded for {order.symbol}")
            factors['daily_volume_violation'] = new_daily_volume / self.risk_limits.max_daily_volume
        elif new_daily_volume > self.risk_limits.max_daily_volume * 0.8:
            warnings.append(f"Approaching daily volume limit for {order.symbol}")
            factors['daily_volume_warning'] = new_daily_volume / self.risk_limits.max_daily_volume
        
        # Order frequency check
        current_minute = datetime.now().strftime("%Y-%m-%d %H:%M")
        orders_this_minute = self.order_counts.get(current_minute, 0)
        
        if orders_this_minute >= self.risk_limits.max_orders_per_minute:
            failed.append("Order frequency limit exceeded")
            factors['frequency_violation'] = orders_this_minute / self.risk_limits.max_orders_per_minute
        elif orders_this_minute >= self.risk_limits.max_orders_per_minute * 0.8:
            warnings.append("High order frequency")
            factors['frequency_warning'] = orders_this_minute / self.risk_limits.max_orders_per_minute
        
        return {'failed': failed, 'warnings': warnings, 'factors': factors}
    
    def _calculate_risk_score(self, risk_factors: Dict[str, float]) -> float:
        """Calculate overall risk score from individual factors"""
        
        if not risk_factors:
            return 0.0
        
        # Weight different risk factors
        weights = {
            'position_limit_violation': 0.3,
            'order_value_violation': 0.2,
            'concentration_violation': 0.25,
            'var_violation': 0.15,
            'daily_volume_violation': 0.1
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for factor, value in risk_factors.items():
            weight = weights.get(factor, 0.05)  # Default small weight
            weighted_score += value * weight
            total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            return min(1.0, weighted_score / total_weight)
        
        return 0.0
    
    def _determine_risk_level(self, risk_score: float, failed_checks: List[str]) -> RiskLevel:
        """Determine risk level based on score and failed checks"""
        
        if failed_checks:
            if risk_score >= 0.8:
                return RiskLevel.CRITICAL
            else:
                return RiskLevel.HIGH
        elif risk_score >= 0.7:
            return RiskLevel.HIGH
        elif risk_score >= 0.3:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _generate_recommendations(
        self,
        order: Order,
        failed_checks: List[str],
        risk_factors: Dict[str, float]
    ) -> Dict[str, Any]:
        """Generate recommendations for risk mitigation"""
        
        recommendations = {}
        
        # Order size recommendations
        if 'order_size_violation' in risk_factors or 'large_order_size' in risk_factors:
            max_recommended_size = int(self.risk_limits.max_order_size * 0.8)
            recommendations['reduce_order_size'] = {
                'current': order.quantity,
                'recommended_max': max_recommended_size,
                'suggestion': 'Consider splitting into multiple orders'
            }
        
        # Position limit recommendations
        if 'position_limit_violation' in risk_factors:
            recommendations['position_management'] = {
                'suggestion': 'Reduce existing position before adding new exposure',
                'alternative': 'Use position-reducing order types'
            }
        
        # Concentration recommendations
        if 'concentration_violation' in risk_factors:
            recommendations['diversification'] = {
                'suggestion': 'Diversify exposure across multiple symbols',
                'alternative': 'Consider sector ETFs for broad exposure'
            }
        
        # Timing recommendations
        if 'frequency_violation' in risk_factors:
            recommendations['timing'] = {
                'suggestion': 'Space out order submissions',
                'recommended_delay': '1-2 minutes between orders'
            }
        
        return recommendations
    
    def _get_estimated_price(self, symbol: str) -> float:
        """Get estimated price for symbol (placeholder)"""
        # In production, would use real-time market data
        return 150.0
    
    def _get_sector(self, symbol: str) -> Optional[str]:
        """Get sector for symbol (placeholder)"""
        # In production, would use symbol-to-sector mapping
        sector_map = {
            'AAPL': 'Technology',
            'MSFT': 'Technology',
            'GOOGL': 'Technology',
            'JPM': 'Financials',
            'BAC': 'Financials'
        }
        return sector_map.get(symbol)
    
    def _get_symbol_volatility(self, symbol: str) -> float:
        """Get symbol volatility estimate (placeholder)"""
        # In production, would calculate from historical data
        return 0.02  # 2% daily volatility
    
    def _estimate_var_impact(self, order: Order) -> float:
        """Estimate VaR impact of order (placeholder)"""
        # Simplified VaR impact calculation
        order_value = order.notional_value
        volatility = self._get_symbol_volatility(order.symbol)
        
        # VaR impact proportional to order size and volatility
        var_impact = (order_value / 1000000) * volatility * 0.01  # Simplified formula
        
        return var_impact
    
    def _estimate_stress_loss(self, order: Order) -> float:
        """Estimate potential stress loss (placeholder)"""
        # Simplified stress test calculation
        order_value = order.notional_value
        stress_factor = 0.20  # 20% stress scenario
        
        potential_loss = order_value * stress_factor / 100000000  # Normalize by $100M portfolio
        
        return potential_loss
    
    def _update_approval_rates(self, result: RiskCheckResult) -> None:
        """Update approval rate tracking"""
        
        risk_level = result.risk_level
        
        if risk_level not in self.approval_rates:
            self.approval_rates[risk_level] = 0.5  # Start at 50%
        
        # Update with exponential moving average
        alpha = 0.1
        new_rate = 1.0 if result.approved else 0.0
        
        self.approval_rates[risk_level] = (
            alpha * new_rate + (1 - alpha) * self.approval_rates[risk_level]
        )
    
    def update_position_cache(self, symbol: str, new_position: int) -> None:
        """Update position cache for risk calculations"""
        self.position_cache[symbol] = new_position
    
    def update_daily_volume(self, symbol: str, volume: int) -> None:
        """Update daily volume tracking"""
        today = datetime.now().date().isoformat()
        key = f"{symbol}_{today}"
        
        if key in self.daily_volumes:
            self.daily_volumes[key] += volume
        else:
            self.daily_volumes[key] = volume
    
    def increment_order_count(self) -> None:
        """Increment order count for frequency tracking"""
        current_minute = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        if current_minute in self.order_counts:
            self.order_counts[current_minute] += 1
        else:
            self.order_counts[current_minute] = 1
        
        # Clean up old entries (keep last hour)
        cutoff_time = datetime.now() - timedelta(hours=1)
        cutoff_key = cutoff_time.strftime("%Y-%m-%d %H:%M")
        
        old_keys = [key for key in self.order_counts.keys() if key < cutoff_key]
        for key in old_keys:
            del self.order_counts[key]
    
    def get_risk_statistics(self) -> Dict[str, Any]:
        """Get risk management statistics"""
        
        avg_check_latency = sum(self.check_latencies) / len(self.check_latencies) if self.check_latencies else 0.0
        
        return {
            'performance': {
                'avg_check_latency_ms': avg_check_latency,
                'total_checks': len(self.check_latencies),
                'max_check_latency_ms': max(self.check_latencies) if self.check_latencies else 0.0
            },
            'approval_rates': self.approval_rates.copy(),
            'current_state': {
                'daily_volumes': dict(list(self.daily_volumes.items())[-10:]),  # Last 10 entries
                'position_cache_size': len(self.position_cache),
                'order_count_entries': len(self.order_counts)
            },
            'risk_limits': {
                'max_order_size': self.risk_limits.max_order_size,
                'max_order_value': self.risk_limits.max_order_value,
                'max_daily_volume': self.risk_limits.max_daily_volume,
                'max_orders_per_minute': self.risk_limits.max_orders_per_minute
            }
        }
    
    def reset_daily_counters(self) -> None:
        """Reset daily volume counters (call at market open)"""
        self.daily_volumes.clear()
        logger.info("Daily volume counters reset")
    
    def update_risk_limits(self, new_limits: Dict[str, Any]) -> None:
        """Update risk limits configuration"""
        
        # Update specific limits
        for key, value in new_limits.items():
            if hasattr(self.risk_limits, key):
                setattr(self.risk_limits, key, value)
        
        logger.info("Risk limits updated", updated_limits=list(new_limits.keys()))