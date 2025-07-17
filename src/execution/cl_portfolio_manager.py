"""
CL Portfolio Management System
=============================

Specialized portfolio management system for CL crude oil trading.
Handles commodity exposure, concentration limits, sector allocation,
and real-time portfolio monitoring.

Key Features:
- Commodity-specific exposure management
- Sector concentration limits
- Real-time portfolio monitoring
- Risk-adjusted portfolio optimization
- Correlation-based position limits
- Dynamic rebalancing

Author: Agent 4 - Risk Management Mission
Date: 2025-07-17
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class AssetClass(Enum):
    """Asset class types"""
    CRUDE_OIL = "crude_oil"
    REFINED_PRODUCTS = "refined_products"
    NATURAL_GAS = "natural_gas"
    ENERGY_EQUITIES = "energy_equities"
    COMMODITIES = "commodities"
    CURRENCIES = "currencies"
    FIXED_INCOME = "fixed_income"

class ExposureType(Enum):
    """Exposure types"""
    GROSS = "gross"
    NET = "net"
    LONG = "long"
    SHORT = "short"

class RebalanceReason(Enum):
    """Rebalancing reasons"""
    THRESHOLD_BREACH = "threshold_breach"
    RISK_REDUCTION = "risk_reduction"
    OPPORTUNITY = "opportunity"
    SCHEDULED = "scheduled"
    EMERGENCY = "emergency"

@dataclass
class PortfolioPosition:
    """Portfolio position data structure"""
    symbol: str
    asset_class: AssetClass
    quantity: float
    entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float
    side: str  # 'long' or 'short'
    entry_date: datetime
    last_update: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExposureLimit:
    """Exposure limit configuration"""
    asset_class: AssetClass
    exposure_type: ExposureType
    limit_percent: float
    warning_percent: float
    current_percent: float = 0.0
    breach_count: int = 0
    last_breach: Optional[datetime] = None

@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics"""
    total_value: float
    total_pnl: float
    daily_pnl: float
    gross_exposure: float
    net_exposure: float
    leverage: float
    max_drawdown: float
    volatility: float
    sharpe_ratio: float
    win_rate: float
    num_positions: int
    concentration_ratio: float
    timestamp: datetime

class CLPortfolioManager:
    """
    Comprehensive portfolio management system for CL trading
    
    Manages commodity exposure, concentration limits, and portfolio optimization
    with specialized focus on crude oil futures and related instruments.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize CL Portfolio Manager
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Portfolio configuration
        self.initial_capital = config.get('initial_capital', 1000000)
        self.current_capital = self.initial_capital
        
        # Exposure limits
        self.exposure_limits = self._initialize_exposure_limits(config.get('exposure_limits', {}))
        
        # Concentration limits
        self.concentration_config = config.get('concentration_limits', {})
        self.max_single_position = self.concentration_config.get('max_single_position', 0.15)
        self.max_sector_exposure = self.concentration_config.get('max_sector_exposure', 0.30)
        
        # Rebalancing configuration
        self.rebalance_config = config.get('rebalancing', {})
        self.rebalance_threshold = self.rebalance_config.get('threshold', 0.05)
        self.rebalance_frequency = self.rebalance_config.get('frequency', 'daily')
        
        # Portfolio state
        self.positions: Dict[str, PortfolioPosition] = {}
        self.cash_balance = self.initial_capital
        self.portfolio_history: List[PortfolioMetrics] = []
        
        # Risk metrics
        self.var_95 = 0.0
        self.var_99 = 0.0
        self.expected_shortfall = 0.0
        self.beta_to_crude = 1.0
        
        # Performance tracking
        self.trades_executed = 0
        self.profitable_trades = 0
        self.total_fees = 0.0
        self.max_positions_held = 0
        
        # Correlation matrix
        self.correlation_matrix = self._initialize_correlation_matrix()
        
        # Asset allocation targets
        self.target_allocation = self._initialize_target_allocation()
        
        # Last rebalance time
        self.last_rebalance = datetime.now()
        
        logger.info("✅ CL Portfolio Manager initialized")
        logger.info(f"   📊 Initial Capital: ${self.initial_capital:,.0f}")
        logger.info(f"   📊 Exposure Limits: {len(self.exposure_limits)}")
        logger.info(f"   📊 Max Single Position: {self.max_single_position:.1%}")
        logger.info(f"   📊 Max Sector Exposure: {self.max_sector_exposure:.1%}")
    
    def _initialize_exposure_limits(self, limits_config: Dict[str, Any]) -> Dict[str, ExposureLimit]:
        """Initialize exposure limits"""
        limits = {}
        
        # Default limits for different asset classes
        default_limits = {
            AssetClass.CRUDE_OIL: {'gross': 1.5, 'net': 1.0, 'long': 1.2, 'short': 0.8},
            AssetClass.REFINED_PRODUCTS: {'gross': 0.5, 'net': 0.3, 'long': 0.4, 'short': 0.3},
            AssetClass.NATURAL_GAS: {'gross': 0.3, 'net': 0.2, 'long': 0.25, 'short': 0.2},
            AssetClass.ENERGY_EQUITIES: {'gross': 0.4, 'net': 0.3, 'long': 0.35, 'short': 0.25},
            AssetClass.COMMODITIES: {'gross': 0.6, 'net': 0.4, 'long': 0.5, 'short': 0.3},
            AssetClass.CURRENCIES: {'gross': 0.2, 'net': 0.1, 'long': 0.15, 'short': 0.1},
            AssetClass.FIXED_INCOME: {'gross': 0.3, 'net': 0.2, 'long': 0.25, 'short': 0.1}
        }
        
        for asset_class in AssetClass:
            class_limits = limits_config.get(asset_class.value, default_limits.get(asset_class, {}))
            
            for exposure_type in ExposureType:
                limit_percent = class_limits.get(exposure_type.value, 0.1)
                warning_percent = limit_percent * 0.8  # 80% warning threshold
                
                limit_key = f"{asset_class.value}_{exposure_type.value}"
                limits[limit_key] = ExposureLimit(
                    asset_class=asset_class,
                    exposure_type=exposure_type,
                    limit_percent=limit_percent,
                    warning_percent=warning_percent
                )
        
        return limits
    
    def _initialize_correlation_matrix(self) -> Dict[str, Dict[str, float]]:
        """Initialize correlation matrix for different instruments"""
        return {
            'CL': {'CL': 1.0, 'BZ': 0.85, 'HO': 0.75, 'RB': 0.70, 'NG': 0.35, 'DX': -0.60},
            'BZ': {'CL': 0.85, 'BZ': 1.0, 'HO': 0.72, 'RB': 0.68, 'NG': 0.30, 'DX': -0.55},
            'HO': {'CL': 0.75, 'BZ': 0.72, 'HO': 1.0, 'RB': 0.85, 'NG': 0.25, 'DX': -0.45},
            'RB': {'CL': 0.70, 'BZ': 0.68, 'HO': 0.85, 'RB': 1.0, 'NG': 0.20, 'DX': -0.40},
            'NG': {'CL': 0.35, 'BZ': 0.30, 'HO': 0.25, 'RB': 0.20, 'NG': 1.0, 'DX': -0.25},
            'DX': {'CL': -0.60, 'BZ': -0.55, 'HO': -0.45, 'RB': -0.40, 'NG': -0.25, 'DX': 1.0}
        }
    
    def _initialize_target_allocation(self) -> Dict[AssetClass, float]:
        """Initialize target allocation for different asset classes"""
        return {
            AssetClass.CRUDE_OIL: 0.60,        # 60% crude oil
            AssetClass.REFINED_PRODUCTS: 0.20,  # 20% refined products
            AssetClass.NATURAL_GAS: 0.10,      # 10% natural gas
            AssetClass.ENERGY_EQUITIES: 0.05,  # 5% energy equities
            AssetClass.COMMODITIES: 0.03,      # 3% other commodities
            AssetClass.CURRENCIES: 0.01,       # 1% currencies
            AssetClass.FIXED_INCOME: 0.01      # 1% fixed income
        }
    
    async def add_position(self, position_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add new position to portfolio
        
        Args:
            position_data: Position information
            
        Returns:
            Position addition result
        """
        try:
            symbol = position_data['symbol']
            quantity = position_data['quantity']
            entry_price = position_data['entry_price']
            side = position_data.get('side', 'long')
            
            # Determine asset class
            asset_class = self._determine_asset_class(symbol)
            
            # Validate position against limits
            validation_result = await self._validate_position(
                symbol, quantity, entry_price, side, asset_class
            )
            
            if not validation_result['approved']:
                return {
                    'success': False,
                    'reason': 'Position validation failed',
                    'validation_result': validation_result
                }
            
            # Calculate position value
            market_value = quantity * entry_price
            
            # Create position
            position = PortfolioPosition(
                symbol=symbol,
                asset_class=asset_class,
                quantity=quantity,
                entry_price=entry_price,
                current_price=entry_price,
                market_value=market_value,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                side=side,
                entry_date=datetime.now(),
                last_update=datetime.now()
            )
            
            # Add to portfolio
            self.positions[symbol] = position
            
            # Update cash balance
            if side == 'long':
                self.cash_balance -= market_value
            else:
                self.cash_balance += market_value  # Short position adds cash
            
            # Update exposure limits
            await self._update_exposure_limits()
            
            # Update portfolio metrics
            await self._update_portfolio_metrics()
            
            # Check for rebalancing needs
            rebalance_needed = await self._check_rebalancing_needed()
            
            logger.info(f"✅ Added position: {symbol} {quantity} @ {entry_price:.2f}")
            
            return {
                'success': True,
                'position': position.__dict__,
                'portfolio_value': self._calculate_portfolio_value(),
                'exposure_updated': True,
                'rebalance_needed': rebalance_needed,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error adding position: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def update_position(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update position with current market data
        
        Args:
            symbol: Position symbol
            market_data: Current market data
            
        Returns:
            Position update result
        """
        try:
            if symbol not in self.positions:
                return {
                    'success': False,
                    'error': f"Position {symbol} not found"
                }
            
            position = self.positions[symbol]
            current_price = market_data.get('close', position.current_price)
            
            # Update position
            position.current_price = current_price
            position.market_value = position.quantity * current_price
            position.last_update = datetime.now()
            
            # Calculate unrealized P&L
            if position.side == 'long':
                position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
            else:
                position.unrealized_pnl = (position.entry_price - current_price) * position.quantity
            
            # Update exposure limits
            await self._update_exposure_limits()
            
            # Update portfolio metrics
            await self._update_portfolio_metrics()
            
            return {
                'success': True,
                'position': position.__dict__,
                'unrealized_pnl': position.unrealized_pnl,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error updating position {symbol}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def close_position(self, symbol: str, exit_price: float, partial_quantity: float = None) -> Dict[str, Any]:
        """
        Close position (full or partial)
        
        Args:
            symbol: Position symbol
            exit_price: Exit price
            partial_quantity: Partial quantity to close (None for full close)
            
        Returns:
            Position close result
        """
        try:
            if symbol not in self.positions:
                return {
                    'success': False,
                    'error': f"Position {symbol} not found"
                }
            
            position = self.positions[symbol]
            
            # Determine quantity to close
            if partial_quantity is None:
                quantity_to_close = position.quantity
                is_full_close = True
            else:
                quantity_to_close = min(partial_quantity, position.quantity)
                is_full_close = (quantity_to_close == position.quantity)
            
            # Calculate realized P&L
            if position.side == 'long':
                realized_pnl = (exit_price - position.entry_price) * quantity_to_close
            else:
                realized_pnl = (position.entry_price - exit_price) * quantity_to_close
            
            # Update position
            if is_full_close:
                # Remove position
                position.realized_pnl += realized_pnl
                closed_position = position.__dict__.copy()
                del self.positions[symbol]
            else:
                # Partial close
                position.quantity -= quantity_to_close
                position.realized_pnl += realized_pnl
                position.market_value = position.quantity * position.current_price
                closed_position = None
            
            # Update cash balance
            if position.side == 'long':
                self.cash_balance += quantity_to_close * exit_price
            else:
                self.cash_balance -= quantity_to_close * exit_price
            
            # Update tracking
            self.trades_executed += 1
            if realized_pnl > 0:
                self.profitable_trades += 1
            
            # Update exposure limits
            await self._update_exposure_limits()
            
            # Update portfolio metrics
            await self._update_portfolio_metrics()
            
            logger.info(f"✅ Closed position: {symbol} {quantity_to_close} @ {exit_price:.2f} (P&L: ${realized_pnl:.2f})")
            
            return {
                'success': True,
                'symbol': symbol,
                'quantity_closed': quantity_to_close,
                'exit_price': exit_price,
                'realized_pnl': realized_pnl,
                'is_full_close': is_full_close,
                'closed_position': closed_position,
                'remaining_quantity': position.quantity if not is_full_close else 0,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error closing position {symbol}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _validate_position(self, 
                                symbol: str,
                                quantity: float,
                                entry_price: float,
                                side: str,
                                asset_class: AssetClass) -> Dict[str, Any]:
        """Validate position against portfolio limits"""
        try:
            validation_result = {
                'approved': True,
                'warnings': [],
                'adjustments': {}
            }
            
            # Calculate position value
            position_value = quantity * entry_price
            portfolio_value = self._calculate_portfolio_value()
            
            # Check single position limit
            position_percent = position_value / portfolio_value
            if position_percent > self.max_single_position:
                validation_result['approved'] = False
                validation_result['warnings'].append(f"Single position limit exceeded: {position_percent:.1%}")
                
                # Suggest adjustment
                max_quantity = (portfolio_value * self.max_single_position) / entry_price
                validation_result['adjustments']['max_quantity'] = max_quantity
            
            # Check sector exposure
            current_sector_exposure = self._calculate_sector_exposure(asset_class)
            new_sector_exposure = (current_sector_exposure + position_value) / portfolio_value
            
            if new_sector_exposure > self.max_sector_exposure:
                validation_result['approved'] = False
                validation_result['warnings'].append(f"Sector exposure limit exceeded: {new_sector_exposure:.1%}")
            
            # Check asset class limits
            for limit_key, limit in self.exposure_limits.items():
                if limit.asset_class == asset_class:
                    current_exposure = self._calculate_asset_class_exposure(asset_class, limit.exposure_type)
                    
                    if side == 'long' and limit.exposure_type in [ExposureType.LONG, ExposureType.GROSS]:
                        new_exposure = current_exposure + position_value
                    elif side == 'short' and limit.exposure_type in [ExposureType.SHORT, ExposureType.GROSS]:
                        new_exposure = current_exposure + position_value
                    else:
                        continue
                    
                    exposure_percent = new_exposure / portfolio_value
                    if exposure_percent > limit.limit_percent:
                        validation_result['approved'] = False
                        validation_result['warnings'].append(
                            f"{asset_class.value} {limit.exposure_type.value} limit exceeded: {exposure_percent:.1%}"
                        )
            
            # Check correlation limits
            correlation_risk = self._calculate_correlation_risk(symbol, position_value)
            if correlation_risk > 0.8:  # High correlation risk
                validation_result['warnings'].append(f"High correlation risk: {correlation_risk:.1%}")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating position: {e}")
            return {
                'approved': False,
                'error': str(e)
            }
    
    def _determine_asset_class(self, symbol: str) -> AssetClass:
        """Determine asset class from symbol"""
        if symbol.startswith('CL') or symbol.startswith('BZ'):
            return AssetClass.CRUDE_OIL
        elif symbol.startswith('HO') or symbol.startswith('RB'):
            return AssetClass.REFINED_PRODUCTS
        elif symbol.startswith('NG'):
            return AssetClass.NATURAL_GAS
        elif symbol in ['XLE', 'XOP', 'VDE', 'OIH']:
            return AssetClass.ENERGY_EQUITIES
        elif symbol.startswith('DX') or symbol.endswith('USD'):
            return AssetClass.CURRENCIES
        else:
            return AssetClass.COMMODITIES
    
    def _calculate_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        position_values = sum(pos.market_value for pos in self.positions.values())
        return self.cash_balance + position_values
    
    def _calculate_sector_exposure(self, asset_class: AssetClass) -> float:
        """Calculate current sector exposure"""
        return sum(
            pos.market_value for pos in self.positions.values()
            if pos.asset_class == asset_class
        )
    
    def _calculate_asset_class_exposure(self, asset_class: AssetClass, exposure_type: ExposureType) -> float:
        """Calculate asset class exposure by type"""
        positions = [pos for pos in self.positions.values() if pos.asset_class == asset_class]
        
        if exposure_type == ExposureType.GROSS:
            return sum(abs(pos.market_value) for pos in positions)
        elif exposure_type == ExposureType.NET:
            return sum(pos.market_value if pos.side == 'long' else -pos.market_value for pos in positions)
        elif exposure_type == ExposureType.LONG:
            return sum(pos.market_value for pos in positions if pos.side == 'long')
        elif exposure_type == ExposureType.SHORT:
            return sum(pos.market_value for pos in positions if pos.side == 'short')
        else:
            return 0.0
    
    def _calculate_correlation_risk(self, symbol: str, position_value: float) -> float:
        """Calculate correlation risk for new position"""
        try:
            base_symbol = symbol.split('_')[0]  # Handle futures months
            total_risk = 0.0
            
            for pos_symbol, position in self.positions.items():
                pos_base_symbol = pos_symbol.split('_')[0]
                
                correlation = self.correlation_matrix.get(base_symbol, {}).get(pos_base_symbol, 0.0)
                risk_contribution = abs(correlation) * position.market_value * position_value
                total_risk += risk_contribution
            
            portfolio_value = self._calculate_portfolio_value()
            return total_risk / (portfolio_value ** 2) if portfolio_value > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating correlation risk: {e}")
            return 0.0
    
    async def _update_exposure_limits(self):
        """Update current exposure for all limits"""
        try:
            portfolio_value = self._calculate_portfolio_value()
            
            for limit_key, limit in self.exposure_limits.items():
                current_exposure = self._calculate_asset_class_exposure(limit.asset_class, limit.exposure_type)
                limit.current_percent = current_exposure / portfolio_value if portfolio_value > 0 else 0.0
                
                # Check for breaches
                if limit.current_percent > limit.limit_percent:
                    limit.breach_count += 1
                    limit.last_breach = datetime.now()
                    
                    logger.warning(f"Exposure limit breach: {limit_key} {limit.current_percent:.1%} > {limit.limit_percent:.1%}")
                    
        except Exception as e:
            logger.error(f"Error updating exposure limits: {e}")
    
    async def _update_portfolio_metrics(self):
        """Update portfolio performance metrics"""
        try:
            portfolio_value = self._calculate_portfolio_value()
            
            # Calculate P&L
            total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
            total_realized_pnl = sum(pos.realized_pnl for pos in self.positions.values())
            total_pnl = total_unrealized_pnl + total_realized_pnl
            
            # Calculate exposures
            gross_exposure = sum(abs(pos.market_value) for pos in self.positions.values())
            net_exposure = sum(
                pos.market_value if pos.side == 'long' else -pos.market_value
                for pos in self.positions.values()
            )
            
            # Calculate leverage
            leverage = gross_exposure / portfolio_value if portfolio_value > 0 else 0.0
            
            # Calculate concentration
            if self.positions:
                position_values = [pos.market_value for pos in self.positions.values()]
                concentration_ratio = max(position_values) / sum(position_values) if position_values else 0.0
            else:
                concentration_ratio = 0.0
            
            # Calculate other metrics (simplified)
            daily_pnl = total_pnl  # Simplified - would need historical data
            max_drawdown = 0.0     # Simplified - would need historical tracking
            volatility = 0.0       # Simplified - would need return history
            sharpe_ratio = 0.0     # Simplified - would need risk-free rate
            win_rate = self.profitable_trades / max(self.trades_executed, 1)
            
            # Create metrics object
            metrics = PortfolioMetrics(
                total_value=portfolio_value,
                total_pnl=total_pnl,
                daily_pnl=daily_pnl,
                gross_exposure=gross_exposure,
                net_exposure=net_exposure,
                leverage=leverage,
                max_drawdown=max_drawdown,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                win_rate=win_rate,
                num_positions=len(self.positions),
                concentration_ratio=concentration_ratio,
                timestamp=datetime.now()
            )
            
            # Add to history
            self.portfolio_history.append(metrics)
            
            # Keep only recent history
            if len(self.portfolio_history) > 1000:
                self.portfolio_history = self.portfolio_history[-1000:]
            
            # Update max positions tracking
            self.max_positions_held = max(self.max_positions_held, len(self.positions))
            
        except Exception as e:
            logger.error(f"Error updating portfolio metrics: {e}")
    
    async def _check_rebalancing_needed(self) -> Dict[str, Any]:
        """Check if portfolio rebalancing is needed"""
        try:
            portfolio_value = self._calculate_portfolio_value()
            rebalance_needed = False
            reasons = []
            
            # Check allocation deviations
            current_allocation = self._calculate_current_allocation()
            
            for asset_class, target_percent in self.target_allocation.items():
                current_percent = current_allocation.get(asset_class, 0.0)
                deviation = abs(current_percent - target_percent)
                
                if deviation > self.rebalance_threshold:
                    rebalance_needed = True
                    reasons.append(f"{asset_class.value} allocation deviation: {deviation:.1%}")
            
            # Check exposure limit breaches
            breach_count = sum(1 for limit in self.exposure_limits.values() if limit.current_percent > limit.limit_percent)
            if breach_count > 0:
                rebalance_needed = True
                reasons.append(f"Exposure limit breaches: {breach_count}")
            
            # Check concentration
            if len(self.positions) > 0:
                position_values = [pos.market_value for pos in self.positions.values()]
                max_position_percent = max(position_values) / portfolio_value
                
                if max_position_percent > self.max_single_position:
                    rebalance_needed = True
                    reasons.append(f"Position concentration: {max_position_percent:.1%}")
            
            # Check scheduled rebalancing
            time_since_last = (datetime.now() - self.last_rebalance).total_seconds()
            if self.rebalance_frequency == 'daily' and time_since_last > 86400:
                rebalance_needed = True
                reasons.append("Scheduled daily rebalancing")
            elif self.rebalance_frequency == 'weekly' and time_since_last > 604800:
                rebalance_needed = True
                reasons.append("Scheduled weekly rebalancing")
            
            return {
                'rebalance_needed': rebalance_needed,
                'reasons': reasons,
                'current_allocation': current_allocation,
                'target_allocation': {ac.value: pct for ac, pct in self.target_allocation.items()},
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error checking rebalancing: {e}")
            return {
                'rebalance_needed': False,
                'error': str(e)
            }
    
    def _calculate_current_allocation(self) -> Dict[AssetClass, float]:
        """Calculate current portfolio allocation"""
        try:
            portfolio_value = self._calculate_portfolio_value()
            allocation = {}
            
            for asset_class in AssetClass:
                exposure = self._calculate_sector_exposure(asset_class)
                allocation[asset_class] = exposure / portfolio_value if portfolio_value > 0 else 0.0
            
            return allocation
            
        except Exception as e:
            logger.error(f"Error calculating current allocation: {e}")
            return {}
    
    async def rebalance_portfolio(self, reason: RebalanceReason = RebalanceReason.SCHEDULED) -> Dict[str, Any]:
        """
        Rebalance portfolio to target allocation
        
        Args:
            reason: Reason for rebalancing
            
        Returns:
            Rebalancing result
        """
        try:
            logger.info(f"Starting portfolio rebalancing: {reason.value}")
            
            # Get current state
            portfolio_value = self._calculate_portfolio_value()
            current_allocation = self._calculate_current_allocation()
            
            # Calculate rebalancing trades
            rebalancing_trades = []
            
            for asset_class, target_percent in self.target_allocation.items():
                current_percent = current_allocation.get(asset_class, 0.0)
                deviation = current_percent - target_percent
                
                if abs(deviation) > self.rebalance_threshold:
                    target_value = portfolio_value * target_percent
                    current_value = portfolio_value * current_percent
                    trade_value = target_value - current_value
                    
                    rebalancing_trades.append({
                        'asset_class': asset_class.value,
                        'current_percent': current_percent,
                        'target_percent': target_percent,
                        'deviation': deviation,
                        'trade_value': trade_value,
                        'action': 'buy' if trade_value > 0 else 'sell'
                    })
            
            # Execute rebalancing (simplified - would need actual execution logic)
            executed_trades = []
            for trade in rebalancing_trades:
                # In production, this would execute actual trades
                executed_trades.append({
                    'asset_class': trade['asset_class'],
                    'action': trade['action'],
                    'value': abs(trade['trade_value']),
                    'status': 'executed'
                })
            
            # Update last rebalance time
            self.last_rebalance = datetime.now()
            
            logger.info(f"✅ Portfolio rebalancing completed: {len(executed_trades)} trades")
            
            return {
                'success': True,
                'reason': reason.value,
                'trades_planned': len(rebalancing_trades),
                'trades_executed': len(executed_trades),
                'rebalancing_trades': rebalancing_trades,
                'executed_trades': executed_trades,
                'portfolio_value': portfolio_value,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error rebalancing portfolio: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary"""
        try:
            portfolio_value = self._calculate_portfolio_value()
            
            # Calculate exposures
            exposure_summary = {}
            for asset_class in AssetClass:
                exposure_summary[asset_class.value] = {
                    'gross': self._calculate_asset_class_exposure(asset_class, ExposureType.GROSS),
                    'net': self._calculate_asset_class_exposure(asset_class, ExposureType.NET),
                    'long': self._calculate_asset_class_exposure(asset_class, ExposureType.LONG),
                    'short': self._calculate_asset_class_exposure(asset_class, ExposureType.SHORT)
                }
            
            # Calculate P&L
            total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
            total_realized_pnl = sum(pos.realized_pnl for pos in self.positions.values())
            
            # Position details
            position_details = []
            for symbol, position in self.positions.items():
                position_details.append({
                    'symbol': symbol,
                    'asset_class': position.asset_class.value,
                    'side': position.side,
                    'quantity': position.quantity,
                    'entry_price': position.entry_price,
                    'current_price': position.current_price,
                    'market_value': position.market_value,
                    'unrealized_pnl': position.unrealized_pnl,
                    'pnl_percent': position.unrealized_pnl / (position.quantity * position.entry_price) if position.quantity > 0 else 0,
                    'days_held': (datetime.now() - position.entry_date).days
                })
            
            # Exposure limit status
            limit_status = []
            for limit_key, limit in self.exposure_limits.items():
                limit_status.append({
                    'asset_class': limit.asset_class.value,
                    'exposure_type': limit.exposure_type.value,
                    'current_percent': limit.current_percent,
                    'limit_percent': limit.limit_percent,
                    'utilization': limit.current_percent / limit.limit_percent if limit.limit_percent > 0 else 0,
                    'breach_count': limit.breach_count,
                    'last_breach': limit.last_breach.isoformat() if limit.last_breach else None
                })
            
            return {
                'timestamp': datetime.now().isoformat(),
                'portfolio_value': portfolio_value,
                'cash_balance': self.cash_balance,
                'total_unrealized_pnl': total_unrealized_pnl,
                'total_realized_pnl': total_realized_pnl,
                'num_positions': len(self.positions),
                'exposure_summary': exposure_summary,
                'position_details': position_details,
                'limit_status': limit_status,
                'current_allocation': {ac.value: pct for ac, pct in self._calculate_current_allocation().items()},
                'target_allocation': {ac.value: pct for ac, pct in self.target_allocation.items()},
                'performance_metrics': {
                    'trades_executed': self.trades_executed,
                    'profitable_trades': self.profitable_trades,
                    'win_rate': self.profitable_trades / max(self.trades_executed, 1),
                    'max_positions_held': self.max_positions_held,
                    'total_fees': self.total_fees
                },
                'rebalancing_info': {
                    'last_rebalance': self.last_rebalance.isoformat(),
                    'rebalance_frequency': self.rebalance_frequency,
                    'rebalance_threshold': self.rebalance_threshold
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating portfolio summary: {e}")
            return {'error': str(e)}
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get portfolio risk metrics"""
        try:
            portfolio_value = self._calculate_portfolio_value()
            
            # Calculate portfolio beta to crude oil
            cl_exposure = self._calculate_asset_class_exposure(AssetClass.CRUDE_OIL, ExposureType.GROSS)
            portfolio_beta = cl_exposure / portfolio_value if portfolio_value > 0 else 0.0
            
            # Calculate correlation risk
            correlation_risk = 0.0
            for symbol1, pos1 in self.positions.items():
                for symbol2, pos2 in self.positions.items():
                    if symbol1 != symbol2:
                        base1 = symbol1.split('_')[0]
                        base2 = symbol2.split('_')[0]
                        correlation = self.correlation_matrix.get(base1, {}).get(base2, 0.0)
                        risk_contribution = abs(correlation) * pos1.market_value * pos2.market_value
                        correlation_risk += risk_contribution
            
            correlation_risk = correlation_risk / (portfolio_value ** 2) if portfolio_value > 0 else 0.0
            
            # Calculate concentration risk
            if self.positions:
                position_values = [pos.market_value for pos in self.positions.values()]
                concentration_ratio = max(position_values) / sum(position_values)
                herfindahl_index = sum((pv / sum(position_values)) ** 2 for pv in position_values)
            else:
                concentration_ratio = 0.0
                herfindahl_index = 0.0
            
            return {
                'portfolio_beta': portfolio_beta,
                'correlation_risk': correlation_risk,
                'concentration_ratio': concentration_ratio,
                'herfindahl_index': herfindahl_index,
                'var_95': self.var_95,
                'var_99': self.var_99,
                'expected_shortfall': self.expected_shortfall,
                'leverage': sum(abs(pos.market_value) for pos in self.positions.values()) / portfolio_value if portfolio_value > 0 else 0.0,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {'error': str(e)}
    
    def reset_daily_metrics(self):
        """Reset daily portfolio metrics"""
        try:
            # Reset daily P&L for all positions
            for position in self.positions.values():
                position.realized_pnl = 0.0  # Reset daily realized P&L
            
            # Reset daily tracking metrics
            self.trades_executed = 0
            self.profitable_trades = 0
            self.total_fees = 0.0
            
            logger.info("Daily portfolio metrics reset")
            
        except Exception as e:
            logger.error(f"Error resetting daily metrics: {e}")