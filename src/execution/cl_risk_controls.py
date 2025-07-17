"""
CL Risk Controls System
======================

Comprehensive risk controls specifically designed for CL crude oil trading.
Implements stop-loss, take-profit, drawdown protection, and circuit breakers
optimized for commodity market characteristics.

Key Features:
- CL-specific stop-loss calculations with volatility adjustment
- Take-profit targets with risk/reward optimization
- Drawdown protection with circuit breakers
- Correlation-based position limits
- Real-time risk monitoring and alerts
- Emergency stop mechanisms

Author: Agent 4 - Risk Management Mission
Date: 2025-07-17
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class RiskControlType(Enum):
    """Types of risk controls"""
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    DRAWDOWN_PROTECTION = "drawdown_protection"
    POSITION_LIMIT = "position_limit"
    CORRELATION_LIMIT = "correlation_limit"
    CIRCUIT_BREAKER = "circuit_breaker"
    EMERGENCY_STOP = "emergency_stop"

class ControlTrigger(Enum):
    """Risk control trigger types"""
    PRICE_BASED = "price_based"
    TIME_BASED = "time_based"
    VOLATILITY_BASED = "volatility_based"
    CORRELATION_BASED = "correlation_based"
    DRAWDOWN_BASED = "drawdown_based"
    MANUAL = "manual"

@dataclass
class RiskControl:
    """Risk control configuration"""
    control_id: str
    control_type: RiskControlType
    trigger_type: ControlTrigger
    enabled: bool = True
    trigger_level: float = 0.0
    action: str = "close_position"
    priority: int = 1
    cooldown_period: int = 300  # seconds
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CLRiskEvent:
    """Risk event data structure"""
    event_id: str
    event_type: str
    symbol: str
    trigger_price: float
    current_price: float
    position_size: float
    risk_amount: float
    timestamp: datetime
    action_taken: str
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

class CLRiskControlSystem:
    """
    Comprehensive risk control system for CL crude oil trading
    
    Implements sophisticated risk controls designed for commodity markets
    including volatility-adjusted stops, correlation limits, and circuit breakers.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize CL Risk Control System
        
        Args:
            config: Risk control configuration
        """
        self.config = config
        
        # CL-specific parameters
        self.cl_contract_size = config.get('cl_contract_size', 1000)
        self.cl_tick_size = config.get('cl_tick_size', 0.01)
        self.cl_tick_value = config.get('cl_tick_value', 10.0)
        
        # Risk control parameters
        self.stop_loss_config = config.get('stop_loss', {})
        self.take_profit_config = config.get('take_profit', {})
        self.drawdown_config = config.get('drawdown_protection', {})
        self.correlation_config = config.get('correlation_limits', {})
        
        # Circuit breaker levels
        self.circuit_breakers = config.get('circuit_breakers', {
            'level_1': 0.10,  # 10% drawdown
            'level_2': 0.15,  # 15% drawdown
            'level_3': 0.20   # 20% drawdown
        })
        
        # Active risk controls
        self.active_controls: Dict[str, RiskControl] = {}
        
        # Risk events history
        self.risk_events: List[CLRiskEvent] = []
        
        # Portfolio state
        self.portfolio_value = config.get('initial_capital', 1000000)
        self.current_drawdown = 0.0
        self.daily_pnl = 0.0
        self.positions = {}
        
        # Performance tracking
        self.controls_triggered = 0
        self.successful_stops = 0
        self.failed_stops = 0
        self.total_risk_saved = 0.0
        
        # Initialize default controls
        self._initialize_default_controls()
        
        logger.info("✅ CL Risk Control System initialized")
        logger.info(f"   📊 Active Controls: {len(self.active_controls)}")
        logger.info(f"   📊 Circuit Breakers: {len(self.circuit_breakers)} levels")
    
    def _initialize_default_controls(self):
        """Initialize default risk controls"""
        try:
            # Stop loss control
            if self.stop_loss_config.get('enabled', True):
                stop_loss_control = RiskControl(
                    control_id="cl_stop_loss",
                    control_type=RiskControlType.STOP_LOSS,
                    trigger_type=ControlTrigger.PRICE_BASED,
                    trigger_level=self.stop_loss_config.get('default_percent', 0.02),
                    action="close_position",
                    priority=1,
                    metadata={
                        'volatility_adjusted': self.stop_loss_config.get('volatility_adjusted', True),
                        'trailing_stop': self.stop_loss_config.get('trailing_stop', True),
                        'max_stop_percent': self.stop_loss_config.get('max_stop_percent', 0.05),
                        'min_stop_percent': self.stop_loss_config.get('min_stop_percent', 0.01)
                    }
                )
                self.active_controls['cl_stop_loss'] = stop_loss_control
            
            # Take profit control
            if self.take_profit_config.get('enabled', True):
                take_profit_control = RiskControl(
                    control_id="cl_take_profit",
                    control_type=RiskControlType.TAKE_PROFIT,
                    trigger_type=ControlTrigger.PRICE_BASED,
                    trigger_level=self.take_profit_config.get('risk_reward_ratio', 1.5),
                    action="close_position",
                    priority=2,
                    metadata={
                        'volatility_adjusted': self.take_profit_config.get('volatility_adjusted', True),
                        'trailing_profit': self.take_profit_config.get('trailing_profit', True),
                        'partial_profit_levels': self.take_profit_config.get('partial_profit_levels', [0.5, 0.75])
                    }
                )
                self.active_controls['cl_take_profit'] = take_profit_control
            
            # Drawdown protection
            if self.drawdown_config.get('enabled', True):
                drawdown_control = RiskControl(
                    control_id="cl_drawdown_protection",
                    control_type=RiskControlType.DRAWDOWN_PROTECTION,
                    trigger_type=ControlTrigger.DRAWDOWN_BASED,
                    trigger_level=self.drawdown_config.get('max_drawdown_percent', 0.20),
                    action="reduce_positions",
                    priority=3,
                    metadata={
                        'daily_loss_limit': self.drawdown_config.get('daily_loss_limit', 0.05),
                        'weekly_loss_limit': self.drawdown_config.get('weekly_loss_limit', 0.10),
                        'monthly_loss_limit': self.drawdown_config.get('monthly_loss_limit', 0.15)
                    }
                )
                self.active_controls['cl_drawdown_protection'] = drawdown_control
            
            # Correlation limits
            if self.correlation_config.get('enabled', True):
                correlation_control = RiskControl(
                    control_id="cl_correlation_limit",
                    control_type=RiskControlType.CORRELATION_LIMIT,
                    trigger_type=ControlTrigger.CORRELATION_BASED,
                    trigger_level=self.correlation_config.get('max_correlation', 0.7),
                    action="reject_trade",
                    priority=4,
                    metadata={
                        'lookback_period': self.correlation_config.get('correlation_lookback', 60),
                        'max_correlated_positions': self.correlation_config.get('max_correlated_positions', 3),
                        'adjustment_factor': self.correlation_config.get('correlation_adjustment_factor', 0.5)
                    }
                )
                self.active_controls['cl_correlation_limit'] = correlation_control
            
            logger.info(f"Initialized {len(self.active_controls)} default risk controls")
            
        except Exception as e:
            logger.error(f"Error initializing default controls: {e}")
    
    async def evaluate_risk_controls(self, 
                                   position_data: Dict[str, Any],
                                   market_data: Dict[str, Any],
                                   portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate all active risk controls
        
        Args:
            position_data: Current position information
            market_data: Current market data
            portfolio_data: Portfolio state
            
        Returns:
            Risk control evaluation results
        """
        control_results = {
            'timestamp': datetime.now().isoformat(),
            'controls_evaluated': 0,
            'controls_triggered': 0,
            'actions_required': [],
            'warnings': [],
            'critical_alerts': []
        }
        
        try:
            # Update portfolio state
            self.portfolio_value = portfolio_data.get('total_value', self.portfolio_value)
            self.current_drawdown = portfolio_data.get('current_drawdown', 0.0)
            self.daily_pnl = portfolio_data.get('daily_pnl', 0.0)
            
            # Evaluate each active control
            for control_id, control in self.active_controls.items():
                if not control.enabled:
                    continue
                
                control_results['controls_evaluated'] += 1
                
                # Check if control is in cooldown
                if self._is_in_cooldown(control):
                    continue
                
                # Evaluate control based on type
                evaluation_result = await self._evaluate_control(
                    control, position_data, market_data, portfolio_data
                )
                
                if evaluation_result['triggered']:
                    control_results['controls_triggered'] += 1
                    
                    # Execute control action
                    action_result = await self._execute_control_action(
                        control, evaluation_result, position_data, market_data
                    )
                    
                    if action_result['success']:
                        control_results['actions_required'].append(action_result)
                        
                        # Record risk event
                        risk_event = CLRiskEvent(
                            event_id=f"risk_{control_id}_{int(datetime.now().timestamp())}",
                            event_type=control.control_type.value,
                            symbol=position_data.get('symbol', 'CL'),
                            trigger_price=evaluation_result.get('trigger_price', 0.0),
                            current_price=market_data.get('close', 0.0),
                            position_size=position_data.get('size', 0.0),
                            risk_amount=evaluation_result.get('risk_amount', 0.0),
                            timestamp=datetime.now(),
                            action_taken=action_result['action'],
                            success=True,
                            metadata=evaluation_result
                        )
                        self.risk_events.append(risk_event)
                        
                        # Update control state
                        control.last_triggered = datetime.now()
                        control.trigger_count += 1
                        
                        # Update performance metrics
                        self.controls_triggered += 1
                        self.successful_stops += 1
                        self.total_risk_saved += evaluation_result.get('risk_amount', 0.0)
                        
                        # Add to critical alerts if high priority
                        if control.priority <= 2:
                            control_results['critical_alerts'].append({
                                'control_id': control_id,
                                'message': f"{control.control_type.value} triggered for {position_data.get('symbol', 'CL')}",
                                'action': action_result['action'],
                                'risk_amount': evaluation_result.get('risk_amount', 0.0)
                            })
                    else:
                        control_results['warnings'].append({
                            'control_id': control_id,
                            'message': f"Failed to execute {control.control_type.value}",
                            'error': action_result.get('error', 'Unknown error')
                        })
                        
                        self.failed_stops += 1
            
            # Check circuit breakers
            circuit_breaker_result = await self._check_circuit_breakers(portfolio_data)
            if circuit_breaker_result['triggered']:
                control_results['critical_alerts'].append(circuit_breaker_result)
            
            # Check emergency conditions
            emergency_result = await self._check_emergency_conditions(
                position_data, market_data, portfolio_data
            )
            if emergency_result['triggered']:
                control_results['critical_alerts'].append(emergency_result)
            
        except Exception as e:
            logger.error(f"Error evaluating risk controls: {e}")
            control_results['warnings'].append({
                'control_id': 'system',
                'message': f"Risk control evaluation error: {e}",
                'error': str(e)
            })
        
        return control_results
    
    async def _evaluate_control(self, 
                              control: RiskControl,
                              position_data: Dict[str, Any],
                              market_data: Dict[str, Any],
                              portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a specific risk control"""
        try:
            if control.control_type == RiskControlType.STOP_LOSS:
                return await self._evaluate_stop_loss(control, position_data, market_data)
            elif control.control_type == RiskControlType.TAKE_PROFIT:
                return await self._evaluate_take_profit(control, position_data, market_data)
            elif control.control_type == RiskControlType.DRAWDOWN_PROTECTION:
                return await self._evaluate_drawdown_protection(control, portfolio_data)
            elif control.control_type == RiskControlType.CORRELATION_LIMIT:
                return await self._evaluate_correlation_limit(control, position_data, portfolio_data)
            else:
                return {'triggered': False, 'reason': 'Unknown control type'}
                
        except Exception as e:
            logger.error(f"Error evaluating control {control.control_id}: {e}")
            return {'triggered': False, 'error': str(e)}
    
    async def _evaluate_stop_loss(self, 
                                control: RiskControl,
                                position_data: Dict[str, Any],
                                market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate stop loss control"""
        try:
            current_price = market_data.get('close', 0.0)
            entry_price = position_data.get('entry_price', 0.0)
            position_side = position_data.get('side', 'long')
            position_size = position_data.get('size', 0.0)
            
            # Calculate stop loss level
            if control.metadata.get('volatility_adjusted', True):
                # Use ATR-based stop loss
                atr = market_data.get('atr_20', 0.0)
                stop_distance = max(2.0 * atr, entry_price * control.trigger_level)
            else:
                # Use fixed percentage stop loss
                stop_distance = entry_price * control.trigger_level
            
            # Apply min/max limits
            max_stop = entry_price * control.metadata.get('max_stop_percent', 0.05)
            min_stop = entry_price * control.metadata.get('min_stop_percent', 0.01)
            stop_distance = max(min(stop_distance, max_stop), min_stop)
            
            # Calculate stop loss price
            if position_side.lower() == 'long':
                stop_loss_price = entry_price - stop_distance
                triggered = current_price <= stop_loss_price
            else:
                stop_loss_price = entry_price + stop_distance
                triggered = current_price >= stop_loss_price
            
            if triggered:
                risk_amount = abs(current_price - entry_price) * position_size * self.cl_contract_size
                
                return {
                    'triggered': True,
                    'trigger_price': stop_loss_price,
                    'current_price': current_price,
                    'entry_price': entry_price,
                    'risk_amount': risk_amount,
                    'stop_distance': stop_distance,
                    'reason': f"Stop loss triggered: {position_side} position stopped at {current_price:.2f}"
                }
            
            return {
                'triggered': False,
                'stop_loss_price': stop_loss_price,
                'current_price': current_price,
                'distance_to_stop': abs(current_price - stop_loss_price) / current_price
            }
            
        except Exception as e:
            logger.error(f"Error evaluating stop loss: {e}")
            return {'triggered': False, 'error': str(e)}
    
    async def _evaluate_take_profit(self, 
                                  control: RiskControl,
                                  position_data: Dict[str, Any],
                                  market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate take profit control"""
        try:
            current_price = market_data.get('close', 0.0)
            entry_price = position_data.get('entry_price', 0.0)
            position_side = position_data.get('side', 'long')
            position_size = position_data.get('size', 0.0)
            
            # Calculate take profit level
            risk_reward_ratio = control.trigger_level
            
            if control.metadata.get('volatility_adjusted', True):
                # Use ATR-based take profit
                atr = market_data.get('atr_20', 0.0)
                stop_distance = max(2.0 * atr, entry_price * 0.02)  # Assume 2% stop
                profit_distance = stop_distance * risk_reward_ratio
            else:
                # Use fixed ratio
                stop_distance = entry_price * 0.02
                profit_distance = stop_distance * risk_reward_ratio
            
            # Calculate take profit price
            if position_side.lower() == 'long':
                take_profit_price = entry_price + profit_distance
                triggered = current_price >= take_profit_price
            else:
                take_profit_price = entry_price - profit_distance
                triggered = current_price <= take_profit_price
            
            if triggered:
                profit_amount = abs(current_price - entry_price) * position_size * self.cl_contract_size
                
                return {
                    'triggered': True,
                    'trigger_price': take_profit_price,
                    'current_price': current_price,
                    'entry_price': entry_price,
                    'profit_amount': profit_amount,
                    'profit_distance': profit_distance,
                    'reason': f"Take profit triggered: {position_side} position closed at {current_price:.2f}"
                }
            
            return {
                'triggered': False,
                'take_profit_price': take_profit_price,
                'current_price': current_price,
                'distance_to_target': abs(take_profit_price - current_price) / current_price
            }
            
        except Exception as e:
            logger.error(f"Error evaluating take profit: {e}")
            return {'triggered': False, 'error': str(e)}
    
    async def _evaluate_drawdown_protection(self, 
                                          control: RiskControl,
                                          portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate drawdown protection control"""
        try:
            current_drawdown = portfolio_data.get('current_drawdown', 0.0)
            daily_loss = portfolio_data.get('daily_loss', 0.0)
            weekly_loss = portfolio_data.get('weekly_loss', 0.0)
            monthly_loss = portfolio_data.get('monthly_loss', 0.0)
            
            # Check various drawdown limits
            max_drawdown = control.trigger_level
            daily_limit = control.metadata.get('daily_loss_limit', 0.05)
            weekly_limit = control.metadata.get('weekly_loss_limit', 0.10)
            monthly_limit = control.metadata.get('monthly_loss_limit', 0.15)
            
            triggered = False
            trigger_reason = ""
            
            if current_drawdown >= max_drawdown:
                triggered = True
                trigger_reason = f"Maximum drawdown exceeded: {current_drawdown:.2%}"
            elif daily_loss >= daily_limit:
                triggered = True
                trigger_reason = f"Daily loss limit exceeded: {daily_loss:.2%}"
            elif weekly_loss >= weekly_limit:
                triggered = True
                trigger_reason = f"Weekly loss limit exceeded: {weekly_loss:.2%}"
            elif monthly_loss >= monthly_limit:
                triggered = True
                trigger_reason = f"Monthly loss limit exceeded: {monthly_loss:.2%}"
            
            if triggered:
                return {
                    'triggered': True,
                    'current_drawdown': current_drawdown,
                    'daily_loss': daily_loss,
                    'weekly_loss': weekly_loss,
                    'monthly_loss': monthly_loss,
                    'trigger_reason': trigger_reason,
                    'risk_amount': self.portfolio_value * current_drawdown
                }
            
            return {
                'triggered': False,
                'current_drawdown': current_drawdown,
                'max_drawdown': max_drawdown,
                'drawdown_utilization': current_drawdown / max_drawdown
            }
            
        except Exception as e:
            logger.error(f"Error evaluating drawdown protection: {e}")
            return {'triggered': False, 'error': str(e)}
    
    async def _evaluate_correlation_limit(self, 
                                        control: RiskControl,
                                        position_data: Dict[str, Any],
                                        portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate correlation limit control"""
        try:
            symbol = position_data.get('symbol', 'CL')
            positions = portfolio_data.get('positions', {})
            
            # Check correlation with existing positions
            max_correlation = control.trigger_level
            high_correlation_positions = []
            
            for pos_symbol, pos_data in positions.items():
                if pos_symbol != symbol:
                    # Calculate correlation (simplified)
                    correlation = self._calculate_position_correlation(symbol, pos_symbol)
                    
                    if correlation > max_correlation:
                        high_correlation_positions.append({
                            'symbol': pos_symbol,
                            'correlation': correlation,
                            'position_value': pos_data.get('value', 0)
                        })
            
            # Check if we exceed maximum correlated positions
            max_correlated = control.metadata.get('max_correlated_positions', 3)
            
            if len(high_correlation_positions) >= max_correlated:
                return {
                    'triggered': True,
                    'high_correlation_positions': high_correlation_positions,
                    'max_correlation': max_correlation,
                    'reason': f"Too many correlated positions: {len(high_correlation_positions)}/{max_correlated}"
                }
            
            return {
                'triggered': False,
                'correlation_count': len(high_correlation_positions),
                'max_allowed': max_correlated
            }
            
        except Exception as e:
            logger.error(f"Error evaluating correlation limit: {e}")
            return {'triggered': False, 'error': str(e)}
    
    def _calculate_position_correlation(self, symbol1: str, symbol2: str) -> float:
        """Calculate correlation between two positions (simplified)"""
        # Simplified correlation calculation
        # In production, this would use historical price correlation
        
        # Oil-related correlations
        oil_symbols = ['CL', 'BZ', 'HO', 'RB', 'UCO', 'USO', 'OIL']
        energy_symbols = ['XLE', 'XOP', 'VDE', 'OIH']
        
        if symbol1 in oil_symbols and symbol2 in oil_symbols:
            return 0.85  # High correlation between oil products
        elif symbol1 in energy_symbols and symbol2 in energy_symbols:
            return 0.75  # High correlation between energy ETFs
        elif (symbol1 in oil_symbols and symbol2 in energy_symbols) or \
             (symbol1 in energy_symbols and symbol2 in oil_symbols):
            return 0.65  # Moderate correlation between oil and energy
        else:
            return 0.3   # Low correlation with other assets
    
    async def _check_circuit_breakers(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check circuit breaker conditions"""
        try:
            current_drawdown = portfolio_data.get('current_drawdown', 0.0)
            
            # Check circuit breaker levels
            for level, threshold in self.circuit_breakers.items():
                if current_drawdown >= threshold:
                    return {
                        'triggered': True,
                        'level': level,
                        'threshold': threshold,
                        'current_drawdown': current_drawdown,
                        'action': self._get_circuit_breaker_action(level),
                        'message': f"Circuit breaker {level} triggered at {current_drawdown:.2%} drawdown"
                    }
            
            return {'triggered': False}
            
        except Exception as e:
            logger.error(f"Error checking circuit breakers: {e}")
            return {'triggered': False, 'error': str(e)}
    
    def _get_circuit_breaker_action(self, level: str) -> str:
        """Get action for circuit breaker level"""
        actions = {
            'level_1': 'reduce_position_sizes_50',
            'level_2': 'reduce_position_sizes_75',
            'level_3': 'stop_all_trading'
        }
        return actions.get(level, 'reduce_position_sizes_50')
    
    async def _check_emergency_conditions(self, 
                                        position_data: Dict[str, Any],
                                        market_data: Dict[str, Any],
                                        portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check emergency conditions"""
        try:
            # Check for extreme volatility
            current_volatility = market_data.get('volatility', 0.0)
            if current_volatility > 0.1:  # 10% volatility threshold
                return {
                    'triggered': True,
                    'condition': 'extreme_volatility',
                    'volatility': current_volatility,
                    'action': 'reduce_positions',
                    'message': f"Extreme volatility detected: {current_volatility:.2%}"
                }
            
            # Check for gap risk
            gap_size = market_data.get('gap_size', 0.0)
            if gap_size > 0.05:  # 5% gap threshold
                return {
                    'triggered': True,
                    'condition': 'large_gap',
                    'gap_size': gap_size,
                    'action': 'emergency_stop',
                    'message': f"Large gap detected: {gap_size:.2%}"
                }
            
            # Check for liquidity crisis
            bid_ask_spread = market_data.get('bid_ask_spread', 0.0)
            if bid_ask_spread > 0.01:  # 1% spread threshold
                return {
                    'triggered': True,
                    'condition': 'liquidity_crisis',
                    'spread': bid_ask_spread,
                    'action': 'suspend_trading',
                    'message': f"Wide bid-ask spread: {bid_ask_spread:.2%}"
                }
            
            return {'triggered': False}
            
        except Exception as e:
            logger.error(f"Error checking emergency conditions: {e}")
            return {'triggered': False, 'error': str(e)}
    
    async def _execute_control_action(self, 
                                    control: RiskControl,
                                    evaluation_result: Dict[str, Any],
                                    position_data: Dict[str, Any],
                                    market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute risk control action"""
        try:
            action = control.action
            
            if action == "close_position":
                return await self._close_position(position_data, market_data, evaluation_result)
            elif action == "reduce_positions":
                return await self._reduce_positions(position_data, market_data, evaluation_result)
            elif action == "reject_trade":
                return await self._reject_trade(position_data, evaluation_result)
            elif action == "emergency_stop":
                return await self._emergency_stop(evaluation_result)
            else:
                return {
                    'success': False,
                    'action': action,
                    'error': f"Unknown action: {action}"
                }
                
        except Exception as e:
            logger.error(f"Error executing control action: {e}")
            return {
                'success': False,
                'action': control.action,
                'error': str(e)
            }
    
    async def _close_position(self, 
                            position_data: Dict[str, Any],
                            market_data: Dict[str, Any],
                            evaluation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Close position due to risk control"""
        try:
            symbol = position_data.get('symbol', 'CL')
            size = position_data.get('size', 0.0)
            current_price = market_data.get('close', 0.0)
            
            # In production, this would send actual close orders
            # For now, we simulate the close
            
            logger.info(f"🛑 Risk control closing position: {symbol} size: {size} at {current_price:.2f}")
            
            return {
                'success': True,
                'action': 'close_position',
                'symbol': symbol,
                'size': size,
                'price': current_price,
                'reason': evaluation_result.get('reason', 'Risk control triggered'),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return {
                'success': False,
                'action': 'close_position',
                'error': str(e)
            }
    
    async def _reduce_positions(self, 
                              position_data: Dict[str, Any],
                              market_data: Dict[str, Any],
                              evaluation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Reduce position sizes due to risk control"""
        try:
            symbol = position_data.get('symbol', 'CL')
            current_size = position_data.get('size', 0.0)
            
            # Reduce position by 50% by default
            reduction_factor = 0.5
            new_size = current_size * (1 - reduction_factor)
            
            logger.info(f"📉 Risk control reducing position: {symbol} from {current_size} to {new_size}")
            
            return {
                'success': True,
                'action': 'reduce_positions',
                'symbol': symbol,
                'original_size': current_size,
                'new_size': new_size,
                'reduction_factor': reduction_factor,
                'reason': evaluation_result.get('reason', 'Risk control triggered'),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error reducing positions: {e}")
            return {
                'success': False,
                'action': 'reduce_positions',
                'error': str(e)
            }
    
    async def _reject_trade(self, 
                          position_data: Dict[str, Any],
                          evaluation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Reject trade due to risk control"""
        try:
            symbol = position_data.get('symbol', 'CL')
            size = position_data.get('size', 0.0)
            
            logger.info(f"❌ Risk control rejecting trade: {symbol} size: {size}")
            
            return {
                'success': True,
                'action': 'reject_trade',
                'symbol': symbol,
                'size': size,
                'reason': evaluation_result.get('reason', 'Risk control triggered'),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error rejecting trade: {e}")
            return {
                'success': False,
                'action': 'reject_trade',
                'error': str(e)
            }
    
    async def _emergency_stop(self, evaluation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute emergency stop"""
        try:
            logger.critical(f"🚨 EMERGENCY STOP TRIGGERED: {evaluation_result.get('reason', 'Unknown')}")
            
            # Disable all controls temporarily
            for control in self.active_controls.values():
                control.enabled = False
            
            return {
                'success': True,
                'action': 'emergency_stop',
                'reason': evaluation_result.get('reason', 'Emergency condition detected'),
                'timestamp': datetime.now().isoformat(),
                'controls_disabled': len(self.active_controls)
            }
            
        except Exception as e:
            logger.error(f"Error executing emergency stop: {e}")
            return {
                'success': False,
                'action': 'emergency_stop',
                'error': str(e)
            }
    
    def _is_in_cooldown(self, control: RiskControl) -> bool:
        """Check if control is in cooldown period"""
        if control.last_triggered is None:
            return False
        
        time_since_trigger = (datetime.now() - control.last_triggered).total_seconds()
        return time_since_trigger < control.cooldown_period
    
    def add_custom_control(self, control: RiskControl):
        """Add custom risk control"""
        try:
            self.active_controls[control.control_id] = control
            logger.info(f"Added custom risk control: {control.control_id}")
        except Exception as e:
            logger.error(f"Error adding custom control: {e}")
    
    def remove_control(self, control_id: str):
        """Remove risk control"""
        try:
            if control_id in self.active_controls:
                del self.active_controls[control_id]
                logger.info(f"Removed risk control: {control_id}")
            else:
                logger.warning(f"Control not found: {control_id}")
        except Exception as e:
            logger.error(f"Error removing control: {e}")
    
    def enable_control(self, control_id: str):
        """Enable risk control"""
        try:
            if control_id in self.active_controls:
                self.active_controls[control_id].enabled = True
                logger.info(f"Enabled risk control: {control_id}")
            else:
                logger.warning(f"Control not found: {control_id}")
        except Exception as e:
            logger.error(f"Error enabling control: {e}")
    
    def disable_control(self, control_id: str):
        """Disable risk control"""
        try:
            if control_id in self.active_controls:
                self.active_controls[control_id].enabled = False
                logger.info(f"Disabled risk control: {control_id}")
            else:
                logger.warning(f"Control not found: {control_id}")
        except Exception as e:
            logger.error(f"Error disabling control: {e}")
    
    def get_control_status(self) -> Dict[str, Any]:
        """Get status of all risk controls"""
        try:
            status = {
                'timestamp': datetime.now().isoformat(),
                'total_controls': len(self.active_controls),
                'enabled_controls': len([c for c in self.active_controls.values() if c.enabled]),
                'controls_in_cooldown': len([c for c in self.active_controls.values() if self._is_in_cooldown(c)]),
                'controls': {}
            }
            
            for control_id, control in self.active_controls.items():
                status['controls'][control_id] = {
                    'type': control.control_type.value,
                    'enabled': control.enabled,
                    'trigger_level': control.trigger_level,
                    'trigger_count': control.trigger_count,
                    'last_triggered': control.last_triggered.isoformat() if control.last_triggered else None,
                    'in_cooldown': self._is_in_cooldown(control)
                }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting control status: {e}")
            return {'error': str(e)}
    
    def get_risk_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent risk events"""
        try:
            recent_events = self.risk_events[-limit:] if limit > 0 else self.risk_events
            return [
                {
                    'event_id': event.event_id,
                    'event_type': event.event_type,
                    'symbol': event.symbol,
                    'trigger_price': event.trigger_price,
                    'current_price': event.current_price,
                    'position_size': event.position_size,
                    'risk_amount': event.risk_amount,
                    'timestamp': event.timestamp.isoformat(),
                    'action_taken': event.action_taken,
                    'success': event.success
                }
                for event in recent_events
            ]
        except Exception as e:
            logger.error(f"Error getting risk events: {e}")
            return []
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get risk control performance metrics"""
        try:
            total_controls = self.controls_triggered + self.failed_stops
            success_rate = self.successful_stops / total_controls if total_controls > 0 else 0
            
            return {
                'controls_triggered': self.controls_triggered,
                'successful_stops': self.successful_stops,
                'failed_stops': self.failed_stops,
                'success_rate': success_rate,
                'total_risk_saved': self.total_risk_saved,
                'average_risk_per_event': self.total_risk_saved / max(self.successful_stops, 1),
                'risk_events_count': len(self.risk_events)
            }
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {}
    
    def reset_daily_metrics(self):
        """Reset daily metrics"""
        try:
            self.daily_pnl = 0.0
            
            # Reset daily trigger counts
            for control in self.active_controls.values():
                if control.last_triggered and \
                   (datetime.now() - control.last_triggered).days >= 1:
                    control.trigger_count = 0
            
            # Clear old risk events (keep last 100)
            if len(self.risk_events) > 100:
                self.risk_events = self.risk_events[-100:]
            
            logger.info("Daily risk control metrics reset")
            
        except Exception as e:
            logger.error(f"Error resetting daily metrics: {e}")