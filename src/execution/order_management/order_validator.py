"""
Order Validator

Ultra-fast order validation with comprehensive risk checks.
Optimized for <50μs validation time to meet <500μs total order placement target.
"""

import logging


import time
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
import structlog

from .order_types import Order, OrderType, OrderSide, TimeInForce, OrderStatus
from ..analytics.market_data import MarketDataProvider

logger = structlog.get_logger()


@dataclass
class ValidationResult:
    """Result of order validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    validation_time_us: float
    risk_score: float
    
    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0
    
    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0


@dataclass
class ValidationConfig:
    """Configuration for order validation"""
    
    # Basic validation
    min_quantity: int = 1
    max_quantity: int = 1000000
    min_price: float = 0.01
    max_price: float = 100000.0
    max_notional: float = 50000000.0  # $50M max
    
    # Symbol validation
    allowed_symbols: Optional[Set[str]] = None
    blocked_symbols: Set[str] = None
    
    # Time validation
    market_open_buffer_minutes: int = 5
    market_close_buffer_minutes: int = 5
    
    # Risk limits
    max_position_size: Dict[str, int] = None
    max_order_value: Dict[str, float] = None
    daily_order_limit: int = 10000
    
    # Performance settings
    enable_market_data_checks: bool = True
    enable_position_checks: bool = True
    enable_risk_checks: bool = True
    
    def __post_init__(self):
        if self.blocked_symbols is None:
            self.blocked_symbols = set()
        if self.max_position_size is None:
            self.max_position_size = {}
        if self.max_order_value is None:
            self.max_order_value = {}


class OrderValidator:
    """
    Ultra-fast order validator for high-frequency execution.
    
    Performs comprehensive validation in <50μs to meet overall
    <500μs order placement latency target.
    """
    
    def __init__(self, config: ValidationConfig, market_data: Optional[MarketDataProvider] = None):
        self.config = config
        self.market_data = market_data
        self.daily_order_counts: Dict[str, int] = {}  # date -> count
        self.validation_cache: Dict[str, ValidationResult] = {}
        
        # Pre-compile validation functions for speed
        self._validation_functions = [
            self._validate_basic_fields,
            self._validate_symbol,
            self._validate_quantity_and_price,
            self._validate_order_type,
            self._validate_time_in_force,
            self._validate_market_hours,
            self._validate_notional_limits,
        ]
        
        # Add optional validations if enabled
        if config.enable_market_data_checks and market_data:
            self._validation_functions.append(self._validate_market_data)
        if config.enable_position_checks:
            self._validation_functions.append(self._validate_position_limits)
        if config.enable_risk_checks:
            self._validation_functions.append(self._validate_risk_limits)
    
    def validate_order(self, order: Order) -> ValidationResult:
        """
        Fast order validation with comprehensive checks.
        Target: <50μs execution time.
        """
        start_time = time.perf_counter()
        
        errors = []
        warnings = []
        risk_score = 0.0
        
        try:
            # Run all validation functions
            for validate_func in self._validation_functions:
                func_errors, func_warnings, func_risk = validate_func(order)
                errors.extend(func_errors)
                warnings.extend(func_warnings)
                risk_score = max(risk_score, func_risk)
                
                # Early exit on critical errors for speed
                if len(errors) > 5:  # Too many errors, stop validation
                    break
            
            # Check daily order limits
            today = datetime.now().date().isoformat()
            daily_count = self.daily_order_counts.get(today, 0)
            if daily_count >= self.config.daily_order_limit:
                errors.append(f"Daily order limit exceeded: {daily_count}/{self.config.daily_order_limit}")
            
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
            logger.error("Order validation exception", error=str(e), order_id=order.order_id)
        
        validation_time = (time.perf_counter() - start_time) * 1_000_000  # Convert to microseconds
        
        result = ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            validation_time_us=validation_time,
            risk_score=risk_score
        )
        
        # Log slow validations
        if validation_time > 50:
            logger.warning(
                "Slow order validation",
                validation_time_us=validation_time,
                order_id=order.order_id,
                symbol=order.symbol
            )
        
        return result
    
    def _validate_basic_fields(self, order: Order) -> Tuple[List[str], List[str], float]:
        """Validate basic order fields"""
        errors = []
        warnings = []
        risk_score = 0.0
        
        # Required fields
        if not order.symbol:
            errors.append("Symbol is required")
        if not order.order_id:
            errors.append("Order ID is required")
        if order.quantity <= 0:
            errors.append("Quantity must be positive")
        
        # Order side validation
        if order.side not in [OrderSide.BUY, OrderSide.SELL]:
            errors.append("Invalid order side")
        
        # Status validation
        if order.status not in [OrderStatus.PENDING]:
            warnings.append(f"Unexpected order status for validation: {order.status}")
        
        return errors, warnings, risk_score
    
    def _validate_symbol(self, order: Order) -> Tuple[List[str], List[str], float]:
        """Validate symbol restrictions"""
        errors = []
        warnings = []
        risk_score = 0.0
        
        symbol = order.symbol.upper()
        
        # Check blocked symbols
        if symbol in self.config.blocked_symbols:
            errors.append(f"Symbol {symbol} is blocked from trading")
            risk_score = 1.0
        
        # Check allowed symbols (if whitelist exists)
        if (self.config.allowed_symbols is not None and 
            symbol not in self.config.allowed_symbols):
            errors.append(f"Symbol {symbol} is not in allowed list")
            risk_score = 0.8
        
        # Basic symbol format validation
        if len(symbol) > 10:
            warnings.append("Symbol unusually long")
        if not symbol.isalnum():
            warnings.append("Symbol contains special characters")
        
        return errors, warnings, risk_score
    
    def _validate_quantity_and_price(self, order: Order) -> Tuple[List[str], List[str], float]:
        """Validate quantity and price ranges"""
        errors = []
        warnings = []
        risk_score = 0.0
        
        # Quantity validation
        if order.quantity < self.config.min_quantity:
            errors.append(f"Quantity {order.quantity} below minimum {self.config.min_quantity}")
        if order.quantity > self.config.max_quantity:
            errors.append(f"Quantity {order.quantity} exceeds maximum {self.config.max_quantity}")
            risk_score = 0.7
        
        # Price validation for limit orders
        if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
            if order.price is None:
                errors.append("Price required for limit orders")
            elif order.price < self.config.min_price:
                errors.append(f"Price {order.price} below minimum {self.config.min_price}")
            elif order.price > self.config.max_price:
                errors.append(f"Price {order.price} exceeds maximum {self.config.max_price}")
                risk_score = 0.6
        
        # Stop price validation
        if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT]:
            if order.stop_price is None:
                errors.append("Stop price required for stop orders")
            elif order.stop_price < self.config.min_price:
                errors.append(f"Stop price {order.stop_price} below minimum {self.config.min_price}")
            elif order.stop_price > self.config.max_price:
                errors.append(f"Stop price {order.stop_price} exceeds maximum {self.config.max_price}")
        
        return errors, warnings, risk_score
    
    def _validate_order_type(self, order: Order) -> Tuple[List[str], List[str], float]:
        """Validate order type specific requirements"""
        errors = []
        warnings = []
        risk_score = 0.0
        
        # Market order validation
        if order.order_type == OrderType.MARKET:
            if order.price is not None:
                warnings.append("Price specified for market order (will be ignored)")
            if order.time_in_force not in [TimeInForce.DAY, TimeInForce.IOC, TimeInForce.FOK]:
                warnings.append("Unusual time in force for market order")
        
        # Limit order validation
        elif order.order_type == OrderType.LIMIT:
            if order.price is None:
                errors.append("Limit price required for limit order")
        
        # Stop order validation
        elif order.order_type == OrderType.STOP:
            if order.stop_price is None:
                errors.append("Stop price required for stop order")
            if order.price is not None:
                warnings.append("Limit price specified for stop order (will be ignored)")
        
        # Stop limit order validation
        elif order.order_type == OrderType.STOP_LIMIT:
            if order.stop_price is None:
                errors.append("Stop price required for stop limit order")
            if order.price is None:
                errors.append("Limit price required for stop limit order")
            if order.price is not None and order.stop_price is not None:
                # Validate price relationship
                if order.side == OrderSide.BUY and order.stop_price < order.price:
                    warnings.append("Buy stop limit: stop price below limit price")
                elif order.side == OrderSide.SELL and order.stop_price > order.price:
                    warnings.append("Sell stop limit: stop price above limit price")
        
        # Algorithm order validation
        elif order.order_type in [OrderType.TWAP, OrderType.VWAP, OrderType.IMPLEMENTATION_SHORTFALL]:
            if not order.algorithm_params:
                warnings.append("Algorithm parameters missing for algo order")
            risk_score = 0.3  # Algo orders have different risk profile
        
        return errors, warnings, risk_score
    
    def _validate_time_in_force(self, order: Order) -> Tuple[List[str], List[str], float]:
        """Validate time in force settings"""
        errors = []
        warnings = []
        risk_score = 0.0
        
        # IOC/FOK only valid for certain order types
        if order.time_in_force == TimeInForce.IOC:
            if order.order_type not in [OrderType.MARKET, OrderType.LIMIT]:
                warnings.append("IOC may not be suitable for this order type")
        
        if order.time_in_force == TimeInForce.FOK:
            if order.order_type != OrderType.LIMIT:
                warnings.append("FOK typically used with limit orders")
        
        # GTD validation
        if order.time_in_force == TimeInForce.GTD:
            if order.expiry_time is None:
                errors.append("Expiry time required for GTD orders")
            elif order.expiry_time <= datetime.now():
                errors.append("Expiry time must be in the future")
        
        return errors, warnings, risk_score
    
    def _validate_market_hours(self, order: Order) -> Tuple[List[str], List[str], float]:
        """Validate market hours and trading sessions"""
        errors = []
        warnings = []
        risk_score = 0.0
        
        now = datetime.now()
        
        # Basic market hours check (US markets: 9:30 AM - 4:00 PM ET)
        # This is simplified - real implementation would use market calendar
        current_hour = now.hour
        
        if current_hour < 9 or current_hour >= 16:
            if order.order_type == OrderType.MARKET:
                warnings.append("Market order submitted outside regular trading hours")
                risk_score = 0.4
            else:
                warnings.append("Order submitted outside regular trading hours")
                risk_score = 0.2
        
        # Market open/close buffer checks
        if current_hour == 9 and now.minute < 30 + self.config.market_open_buffer_minutes:
            warnings.append("Order near market open - increased volatility expected")
            risk_score = 0.3
        
        if current_hour == 15 and now.minute >= 60 - self.config.market_close_buffer_minutes:
            warnings.append("Order near market close - increased volatility expected")
            risk_score = 0.3
        
        return errors, warnings, risk_score
    
    def _validate_notional_limits(self, order: Order) -> Tuple[List[str], List[str], float]:
        """Validate notional value limits"""
        errors = []
        warnings = []
        risk_score = 0.0
        
        # Calculate notional value
        if order.price is not None:
            notional = abs(order.quantity * order.price)
        else:
            # For market orders, we need an estimate
            # In production, this would use current market price
            if self.market_data:
                try:
                    current_price = self.market_data.get_current_price(order.symbol)
                    notional = abs(order.quantity * current_price)
                except (ValueError, TypeError, AttributeError, KeyError) as e:
                    notional = 0.0  # Can't calculate without price
            else:
                notional = 0.0
        
        # Check against limits
        if notional > self.config.max_notional:
            errors.append(f"Order notional ${notional:,.0f} exceeds limit ${self.config.max_notional:,.0f}")
            risk_score = 0.9
        elif notional > self.config.max_notional * 0.8:
            warnings.append(f"Large order notional: ${notional:,.0f}")
            risk_score = 0.5
        
        # Symbol-specific limits
        symbol_limit = self.config.max_order_value.get(order.symbol)
        if symbol_limit and notional > symbol_limit:
            errors.append(f"Order exceeds symbol-specific limit for {order.symbol}: ${notional:,.0f} > ${symbol_limit:,.0f}")
            risk_score = 0.8
        
        return errors, warnings, risk_score
    
    def _validate_market_data(self, order: Order) -> Tuple[List[str], List[str], float]:
        """Validate against current market data"""
        errors = []
        warnings = []
        risk_score = 0.0
        
        if not self.market_data:
            return errors, warnings, risk_score
        
        try:
            # Get current market data
            current_price = self.market_data.get_current_price(order.symbol)
            spread = self.market_data.get_spread(order.symbol)
            volume = self.market_data.get_volume(order.symbol)
            
            # Price reasonableness check
            if order.price is not None and current_price > 0:
                price_deviation = abs(order.price - current_price) / current_price
                if price_deviation > 0.10:  # 10% deviation
                    warnings.append(f"Order price {price_deviation:.1%} from market")
                    risk_score = max(risk_score, 0.6)
                elif price_deviation > 0.05:  # 5% deviation
                    warnings.append(f"Order price {price_deviation:.1%} from market")
                    risk_score = max(risk_score, 0.3)
            
            # Market impact estimate
            if volume > 0:
                volume_participation = order.quantity / volume
                if volume_participation > 0.20:  # >20% of volume
                    warnings.append(f"Order size {volume_participation:.1%} of recent volume")
                    risk_score = max(risk_score, 0.7)
                elif volume_participation > 0.10:  # >10% of volume
                    warnings.append(f"Large order relative to volume: {volume_participation:.1%}")
                    risk_score = max(risk_score, 0.4)
            
            # Spread check
            if spread > 0 and order.order_type == OrderType.MARKET:
                if spread / current_price > 0.005:  # Wide spread (>0.5%)
                    warnings.append(f"Wide spread {spread/current_price:.2%} for market order")
                    risk_score = max(risk_score, 0.4)
        
        except Exception as e:
            warnings.append(f"Market data validation failed: {str(e)}")
        
        return errors, warnings, risk_score
    
    def _validate_position_limits(self, order: Order) -> Tuple[List[str], List[str], float]:
        """Validate position size limits"""
        errors = []
        warnings = []
        risk_score = 0.0
        
        # Get position limit for symbol
        position_limit = self.config.max_position_size.get(
            order.symbol,
            self.config.max_position_size.get('default', float('inf'))
        )
        
        # This would check current position + order quantity in production
        # For now, just validate order quantity against limit
        if abs(order.quantity) > position_limit:
            errors.append(f"Order quantity {order.quantity} exceeds position limit {position_limit}")
            risk_score = 0.9
        elif abs(order.quantity) > position_limit * 0.8:
            warnings.append(f"Large order relative to position limit")
            risk_score = 0.5
        
        return errors, warnings, risk_score
    
    def _validate_risk_limits(self, order: Order) -> Tuple[List[str], List[str], float]:
        """Validate additional risk constraints"""
        errors = []
        warnings = []
        risk_score = 0.0
        
        # Check for high-risk symbols (this would be configurable)
        high_risk_symbols = {'TSLA', 'GME', 'AMC', 'NVDA'}  # Example
        if order.symbol in high_risk_symbols:
            warnings.append(f"High volatility symbol: {order.symbol}")
            risk_score = max(risk_score, 0.5)
        
        # Check for high-risk order types
        if order.order_type in [OrderType.MARKET]:
            if order.quantity > 10000:  # Large market order
                warnings.append("Large market order - high market impact risk")
                risk_score = max(risk_score, 0.6)
        
        # Check priority levels
        if order.priority.value >= 4:  # URGENT priority
            warnings.append("Urgent priority order - review execution strategy")
            risk_score = max(risk_score, 0.3)
        
        return errors, warnings, risk_score
    
    def validate_batch(self, orders: List[Order]) -> List[ValidationResult]:
        """Validate multiple orders efficiently"""
        results = []
        
        for order in orders:
            result = self.validate_order(order)
            results.append(result)
            
            # Update daily order count if valid
            if result.is_valid:
                today = datetime.now().date().isoformat()
                self.daily_order_counts[today] = self.daily_order_counts.get(today, 0) + 1
        
        return results
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation performance statistics"""
        return {
            'cache_size': len(self.validation_cache),
            'daily_order_counts': self.daily_order_counts.copy(),
            'config': {
                'min_quantity': self.config.min_quantity,
                'max_quantity': self.config.max_quantity,
                'max_notional': self.config.max_notional,
                'daily_order_limit': self.config.daily_order_limit,
                'blocked_symbols_count': len(self.config.blocked_symbols),
                'allowed_symbols_count': len(self.config.allowed_symbols) if self.config.allowed_symbols else None
            }
        }
    
    def reset_daily_counts(self) -> None:
        """Reset daily order counts (call at market open)"""
        self.daily_order_counts.clear()
    
    def update_config(self, new_config: ValidationConfig) -> None:
        """Update validation configuration"""
        self.config = new_config
        self.validation_cache.clear()  # Clear cache when config changes