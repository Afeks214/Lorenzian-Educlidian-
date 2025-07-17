"""
CL-Specific Risk Management System
=================================

Comprehensive risk management system specifically designed for CL crude oil trading.
Implements volatility-based position sizing, commodity-specific risk controls,
and real-time monitoring for oil futures trading.

Key Features:
- CL-specific volatility-based position sizing using ATR
- Commodity risk controls with oil market considerations
- Dynamic position sizing based on market conditions
- Portfolio management for energy sector exposure
- Real-time risk monitoring and alerts
- Execution controls with realistic market impact modeling

Author: Agent 4 - Risk Management Mission
Date: 2025-07-17
"""

import asyncio
import logging
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import deque
import json
import warnings
from enum import Enum

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Risk severity levels for alerting"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class MarketCondition(Enum):
    """Market condition classification"""
    TRENDING = "trending"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    GEOPOLITICAL_RISK = "geopolitical_risk"

@dataclass
class CLRiskMetrics:
    """CL-specific risk metrics"""
    atr_20: float = 0.0
    atr_50: float = 0.0
    volatility_percentile: float = 0.0
    market_condition: MarketCondition = MarketCondition.RANGING
    inventory_impact: float = 0.0
    geopolitical_risk_score: float = 0.0
    session_liquidity: float = 0.0
    overnight_gap_risk: float = 0.0
    correlation_to_usd: float = 0.0
    correlation_to_equities: float = 0.0

@dataclass
class CLPosition:
    """CL position data structure"""
    symbol: str
    side: str  # 'long' or 'short'
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    entry_timestamp: datetime = field(default_factory=datetime.now)
    risk_amount: float = 0.0
    position_value: float = 0.0

@dataclass
class CLRiskAlert:
    """Risk alert data structure"""
    alert_id: str
    level: RiskLevel
    message: str
    timestamp: datetime
    metrics: Dict[str, Any]
    action_required: bool = False
    escalation_time: Optional[datetime] = None

class CLRiskManager:
    """
    Comprehensive CL-specific risk management system
    
    Implements sophisticated risk management specifically designed for crude oil trading,
    including volatility-based position sizing, commodity-specific risk controls,
    and real-time monitoring.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize CL Risk Manager
        
        Args:
            config: Risk management configuration
        """
        self.config = config
        
        # Position sizing parameters
        self.max_risk_per_trade = config.get('max_risk_per_trade', 0.02)  # 2%
        self.max_position_size = config.get('max_position_size', 0.10)    # 10%
        self.min_position_size = config.get('min_position_size', 0.005)   # 0.5%
        self.volatility_lookback = config.get('volatility_lookback', 20)
        
        # CL-specific parameters
        self.cl_contract_size = config.get('cl_contract_size', 1000)  # 1000 barrels per contract
        self.cl_tick_size = config.get('cl_tick_size', 0.01)          # $0.01 per barrel
        self.cl_tick_value = config.get('cl_tick_value', 10.0)        # $10 per tick
        
        # Risk controls
        self.max_drawdown = config.get('max_drawdown', 0.20)          # 20%
        self.daily_loss_limit = config.get('daily_loss_limit', 0.05)  # 5%
        self.leverage_limit = config.get('leverage_limit', 3.0)       # 3:1
        
        # Market condition adjustments
        self.market_adjustments = config.get('market_condition_adjustments', {})
        
        # Portfolio state
        self.portfolio_value = config.get('initial_capital', 1000000)
        self.positions: Dict[str, CLPosition] = {}
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.max_drawdown_reached = 0.0
        
        # Risk monitoring
        self.risk_metrics_history = deque(maxlen=1000)
        self.alerts = deque(maxlen=100)
        self.risk_breaches = deque(maxlen=50)
        
        # Performance tracking
        self.trades_executed = 0
        self.trades_stopped = 0
        self.risk_adjusted_returns = []
        
        # Market data cache
        self.price_history = deque(maxlen=250)  # ~1 year of daily data
        self.volume_history = deque(maxlen=100)
        self.volatility_cache = {}
        
        logger.info("✅ CL Risk Manager initialized")
        logger.info(f"   📊 Max Risk Per Trade: {self.max_risk_per_trade:.1%}")
        logger.info(f"   📊 Max Position Size: {self.max_position_size:.1%}")
        logger.info(f"   📊 Max Drawdown: {self.max_drawdown:.1%}")
        logger.info(f"   📊 Daily Loss Limit: {self.daily_loss_limit:.1%}")
        
    async def calculate_position_size(self, 
                                    signal_data: Dict[str, Any],
                                    market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate optimal position size for CL trade using volatility-based sizing
        
        Args:
            signal_data: Trading signal information
            market_data: Current market data
            
        Returns:
            Position sizing recommendation
        """
        try:
            # Extract signal parameters
            confidence = signal_data.get('confidence', 0.5)
            signal_strength = signal_data.get('strength', 0.5)
            direction = signal_data.get('direction', 'long')
            
            # Calculate CL-specific risk metrics
            risk_metrics = await self._calculate_cl_risk_metrics(market_data)
            
            # Base position size using volatility adjustment
            base_size = self._calculate_volatility_based_size(risk_metrics)
            
            # Apply confidence adjustment
            confidence_adjusted_size = base_size * min(confidence, 1.0)
            
            # Apply market condition adjustment
            market_adjusted_size = self._apply_market_condition_adjustment(
                confidence_adjusted_size, risk_metrics.market_condition
            )
            
            # Apply Kelly Criterion if enabled
            if self.config.get('kelly_criterion', {}).get('enabled', False):
                kelly_size = self._calculate_kelly_position_size(signal_data, risk_metrics)
                kelly_adjusted_size = min(market_adjusted_size, kelly_size)
            else:
                kelly_adjusted_size = market_adjusted_size
            
            # Apply position limits
            final_size = self._apply_position_limits(kelly_adjusted_size, risk_metrics)
            
            # Calculate risk amount
            entry_price = market_data.get('close', 0)
            stop_loss_price = self._calculate_stop_loss(entry_price, direction, risk_metrics)
            risk_per_contract = abs(entry_price - stop_loss_price) * self.cl_contract_size
            risk_amount = final_size * risk_per_contract
            
            return {
                'recommended_size': final_size,
                'risk_amount': risk_amount,
                'risk_percentage': risk_amount / self.portfolio_value,
                'stop_loss_price': stop_loss_price,
                'sizing_factors': {
                    'base_size': base_size,
                    'confidence_adjustment': confidence,
                    'market_condition': risk_metrics.market_condition.value,
                    'volatility_adjustment': risk_metrics.atr_20,
                    'kelly_fraction': kelly_size if 'kelly_size' in locals() else None
                },
                'risk_metrics': risk_metrics.__dict__,
                'approved': risk_amount <= (self.portfolio_value * self.max_risk_per_trade)
            }
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return {
                'recommended_size': 0,
                'risk_amount': 0,
                'risk_percentage': 0,
                'approved': False,
                'error': str(e)
            }
    
    async def _calculate_cl_risk_metrics(self, market_data: Dict[str, Any]) -> CLRiskMetrics:
        """Calculate CL-specific risk metrics"""
        try:
            # Extract price data
            prices = market_data.get('prices', [])
            if len(prices) < 20:
                return CLRiskMetrics()
            
            # Calculate ATR (Average True Range)
            atr_20 = self._calculate_atr(prices[-20:])
            atr_50 = self._calculate_atr(prices[-50:]) if len(prices) >= 50 else atr_20
            
            # Calculate volatility percentile
            recent_volatility = np.std([p['close'] for p in prices[-20:]]) if len(prices) >= 20 else 0
            historical_volatilities = [np.std([p['close'] for p in prices[i:i+20]]) 
                                     for i in range(0, len(prices)-20, 5)]
            volatility_percentile = np.percentile(historical_volatilities, 50) if historical_volatilities else 0
            
            # Determine market condition
            market_condition = self._classify_market_condition(prices, atr_20)
            
            # Calculate inventory impact (simplified)
            inventory_impact = self._calculate_inventory_impact(market_data)
            
            # Calculate geopolitical risk score
            geopolitical_risk = self._calculate_geopolitical_risk(market_data)
            
            # Calculate session liquidity
            session_liquidity = self._calculate_session_liquidity(market_data)
            
            # Calculate overnight gap risk
            overnight_gap_risk = self._calculate_overnight_gap_risk(prices)
            
            return CLRiskMetrics(
                atr_20=atr_20,
                atr_50=atr_50,
                volatility_percentile=volatility_percentile,
                market_condition=market_condition,
                inventory_impact=inventory_impact,
                geopolitical_risk_score=geopolitical_risk,
                session_liquidity=session_liquidity,
                overnight_gap_risk=overnight_gap_risk
            )
            
        except Exception as e:
            logger.error(f"Error calculating CL risk metrics: {e}")
            return CLRiskMetrics()
    
    def _calculate_atr(self, prices: List[Dict[str, float]]) -> float:
        """Calculate Average True Range for CL"""
        if len(prices) < 2:
            return 0.0
        
        true_ranges = []
        for i in range(1, len(prices)):
            high = prices[i]['high']
            low = prices[i]['low']
            prev_close = prices[i-1]['close']
            
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            true_ranges.append(tr)
        
        return np.mean(true_ranges) if true_ranges else 0.0
    
    def _classify_market_condition(self, prices: List[Dict[str, float]], atr: float) -> MarketCondition:
        """Classify current market condition"""
        if len(prices) < 20:
            return MarketCondition.RANGING
        
        # Get recent price data
        recent_closes = [p['close'] for p in prices[-20:]]
        
        # Calculate trend strength
        trend_strength = (recent_closes[-1] - recent_closes[0]) / recent_closes[0]
        
        # Calculate volatility
        volatility = np.std(recent_closes) / np.mean(recent_closes)
        
        # Classify market condition
        if abs(trend_strength) > 0.05:  # Strong trend
            return MarketCondition.TRENDING
        elif volatility > 0.03:  # High volatility
            return MarketCondition.HIGH_VOLATILITY
        elif volatility < 0.01:  # Low volatility
            return MarketCondition.LOW_VOLATILITY
        else:
            return MarketCondition.RANGING
    
    def _calculate_inventory_impact(self, market_data: Dict[str, Any]) -> float:
        """Calculate impact of oil inventory reports"""
        # Simplified inventory impact calculation
        # In production, this would use actual EIA/API inventory data
        inventory_change = market_data.get('inventory_change', 0)
        
        # Normalize inventory impact (-1 to 1)
        # Negative inventory change (drawdown) is bullish
        # Positive inventory change (build) is bearish
        normalized_impact = np.tanh(inventory_change / 1000000)  # Normalize by 1M barrels
        
        return -normalized_impact  # Invert for price impact
    
    def _calculate_geopolitical_risk(self, market_data: Dict[str, Any]) -> float:
        """Calculate geopolitical risk score"""
        # Simplified geopolitical risk calculation
        # In production, this would use news sentiment analysis and geopolitical event data
        
        # Check for elevated volatility as proxy for geopolitical risk
        prices = market_data.get('prices', [])
        if len(prices) < 10:
            return 0.0
        
        recent_volatility = np.std([p['close'] for p in prices[-10:]])
        historical_volatility = np.std([p['close'] for p in prices[-50:]]) if len(prices) >= 50 else recent_volatility
        
        # Risk score based on volatility spike
        volatility_ratio = recent_volatility / (historical_volatility + 1e-8)
        risk_score = min(volatility_ratio - 1.0, 1.0)  # Cap at 1.0
        
        return max(risk_score, 0.0)  # Ensure non-negative
    
    def _calculate_session_liquidity(self, market_data: Dict[str, Any]) -> float:
        """Calculate session liquidity for CL"""
        # CL liquidity varies by session
        # ETH (Electronic Trading Hours): Lower liquidity
        # RTH (Regular Trading Hours): Higher liquidity
        
        current_hour = datetime.now().hour
        
        # ETH sessions (typically 6 PM - 5 PM ET next day)
        # Lower liquidity during Asian and European sessions
        if 0 <= current_hour <= 4 or 18 <= current_hour <= 23:
            return 0.6  # Lower liquidity
        else:
            return 1.0  # Higher liquidity during US session
    
    def _calculate_overnight_gap_risk(self, prices: List[Dict[str, float]]) -> float:
        """Calculate overnight gap risk for CL"""
        if len(prices) < 10:
            return 0.0
        
        # Calculate gap sizes over recent periods
        gaps = []
        for i in range(1, min(len(prices), 20)):
            gap = abs(prices[i]['open'] - prices[i-1]['close']) / prices[i-1]['close']
            gaps.append(gap)
        
        # Return average gap size as risk measure
        return np.mean(gaps) if gaps else 0.0
    
    def _calculate_volatility_based_size(self, risk_metrics: CLRiskMetrics) -> float:
        """Calculate position size based on volatility"""
        if risk_metrics.atr_20 <= 0:
            return self.min_position_size
        
        # Base risk amount
        base_risk_amount = self.portfolio_value * self.max_risk_per_trade
        
        # Calculate position size based on ATR
        # Higher ATR = smaller position size
        atr_adjustment = 1.0 / (1.0 + risk_metrics.atr_20)
        
        # Calculate contracts based on risk
        # Assuming stop loss at 2 * ATR
        stop_distance = 2.0 * risk_metrics.atr_20
        risk_per_contract = stop_distance * self.cl_contract_size
        
        if risk_per_contract > 0:
            contracts = base_risk_amount / risk_per_contract
            return min(contracts, self.max_position_size * self.portfolio_value / (risk_per_contract * self.cl_contract_size))
        else:
            return self.min_position_size
    
    def _apply_market_condition_adjustment(self, base_size: float, market_condition: MarketCondition) -> float:
        """Apply market condition adjustments to position size"""
        adjustments = self.market_adjustments
        
        if market_condition == MarketCondition.TRENDING:
            multiplier = adjustments.get('trending_market_multiplier', 1.2)
        elif market_condition == MarketCondition.RANGING:
            multiplier = adjustments.get('ranging_market_multiplier', 0.8)
        elif market_condition == MarketCondition.HIGH_VOLATILITY:
            multiplier = adjustments.get('high_volatility_multiplier', 0.7)
        elif market_condition == MarketCondition.LOW_VOLATILITY:
            multiplier = adjustments.get('low_volatility_multiplier', 1.1)
        else:
            multiplier = 1.0
        
        return base_size * multiplier
    
    def _calculate_kelly_position_size(self, signal_data: Dict[str, Any], risk_metrics: CLRiskMetrics) -> float:
        """Calculate Kelly Criterion position size"""
        kelly_config = self.config.get('kelly_criterion', {})
        if not kelly_config.get('enabled', False):
            return float('inf')  # No Kelly limit
        
        # Get signal statistics
        win_rate = signal_data.get('win_rate', 0.5)
        avg_win = signal_data.get('avg_win', 0.02)
        avg_loss = signal_data.get('avg_loss', 0.01)
        
        # Calculate Kelly fraction
        if avg_loss > 0:
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_loss
        else:
            kelly_fraction = 0.0
        
        # Apply safety factor and maximum fraction
        safety_factor = kelly_config.get('safety_factor', 0.5)
        max_kelly_fraction = kelly_config.get('max_kelly_fraction', 0.25)
        
        adjusted_kelly = min(kelly_fraction * safety_factor, max_kelly_fraction)
        
        # Convert to position size
        kelly_risk_amount = self.portfolio_value * adjusted_kelly
        
        # Calculate contracts
        stop_distance = 2.0 * risk_metrics.atr_20
        risk_per_contract = stop_distance * self.cl_contract_size
        
        if risk_per_contract > 0:
            return kelly_risk_amount / risk_per_contract
        else:
            return 0.0
    
    def _apply_position_limits(self, calculated_size: float, risk_metrics: CLRiskMetrics) -> float:
        """Apply position size limits"""
        # Apply minimum and maximum limits
        size = max(calculated_size, self.min_position_size)
        size = min(size, self.max_position_size)
        
        # Apply concentration limits
        current_cl_exposure = sum(pos.quantity for pos in self.positions.values() if pos.symbol.startswith('CL'))
        max_cl_exposure = self.portfolio_value * 0.5  # Max 50% in CL
        
        if (current_cl_exposure + size) * self.cl_contract_size > max_cl_exposure:
            size = max(0, (max_cl_exposure - current_cl_exposure * self.cl_contract_size) / self.cl_contract_size)
        
        return size
    
    def _calculate_stop_loss(self, entry_price: float, direction: str, risk_metrics: CLRiskMetrics) -> float:
        """Calculate stop loss price for CL"""
        # Use 2x ATR for stop loss
        stop_distance = 2.0 * risk_metrics.atr_20
        
        if direction.lower() == 'long':
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance
    
    async def validate_trade(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate trade against CL-specific risk limits
        
        Args:
            trade_data: Trade information
            
        Returns:
            Validation result
        """
        validation_result = {
            'approved': True,
            'adjustments': {},
            'risk_checks': {},
            'warnings': [],
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Position size validation
            size_check = await self._validate_position_size(trade_data)
            validation_result['risk_checks']['position_size'] = size_check
            
            if not size_check['passed']:
                validation_result['approved'] = False
                validation_result['adjustments'].update(size_check['adjustments'])
            
            # Risk amount validation
            risk_check = await self._validate_risk_amount(trade_data)
            validation_result['risk_checks']['risk_amount'] = risk_check
            
            if not risk_check['passed']:
                validation_result['approved'] = False
                validation_result['warnings'].append(risk_check['message'])
            
            # Drawdown validation
            drawdown_check = await self._validate_drawdown_limits()
            validation_result['risk_checks']['drawdown'] = drawdown_check
            
            if not drawdown_check['passed']:
                validation_result['approved'] = False
                validation_result['warnings'].append("Drawdown limit exceeded")
            
            # Leverage validation
            leverage_check = await self._validate_leverage_limits(trade_data)
            validation_result['risk_checks']['leverage'] = leverage_check
            
            if not leverage_check['passed']:
                validation_result['approved'] = False
                validation_result['warnings'].append("Leverage limit exceeded")
            
            # Market condition validation
            market_check = await self._validate_market_conditions(trade_data)
            validation_result['risk_checks']['market_conditions'] = market_check
            
            if not market_check['passed']:
                validation_result['warnings'].append(market_check['message'])
            
            # Log validation result
            if validation_result['approved']:
                logger.info(f"✅ Trade validated: {trade_data.get('symbol', 'UNKNOWN')}")
            else:
                logger.warning(f"❌ Trade rejected: {trade_data.get('symbol', 'UNKNOWN')}")
                self._record_risk_breach(trade_data, validation_result)
            
        except Exception as e:
            logger.error(f"Error validating trade: {e}")
            validation_result['approved'] = False
            validation_result['warnings'].append(f"Validation error: {e}")
        
        return validation_result
    
    async def _validate_position_size(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate position size limits"""
        try:
            size = trade_data.get('size', 0)
            symbol = trade_data.get('symbol', '')
            
            # Check maximum position size
            position_value = size * self.cl_contract_size * trade_data.get('price', 0)
            max_position_value = self.portfolio_value * self.max_position_size
            
            if position_value <= max_position_value:
                return {
                    'passed': True,
                    'position_value': position_value,
                    'limit': max_position_value,
                    'adjustments': {}
                }
            else:
                # Calculate adjusted size
                adjusted_size = max_position_value / (self.cl_contract_size * trade_data.get('price', 1))
                
                return {
                    'passed': False,
                    'position_value': position_value,
                    'limit': max_position_value,
                    'adjustments': {
                        'size': adjusted_size,
                        'reason': 'Position size limit exceeded'
                    }
                }
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'adjustments': {'size': 0, 'reason': 'Position size validation failed'}
            }
    
    async def _validate_risk_amount(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate risk amount limits"""
        try:
            size = trade_data.get('size', 0)
            entry_price = trade_data.get('price', 0)
            stop_loss = trade_data.get('stop_loss', 0)
            
            if stop_loss > 0:
                risk_per_contract = abs(entry_price - stop_loss) * self.cl_contract_size
                total_risk = size * risk_per_contract
                max_risk = self.portfolio_value * self.max_risk_per_trade
                
                return {
                    'passed': total_risk <= max_risk,
                    'risk_amount': total_risk,
                    'limit': max_risk,
                    'message': f"Risk amount: ${total_risk:.2f} (limit: ${max_risk:.2f})"
                }
            else:
                return {
                    'passed': True,
                    'message': "No stop loss specified"
                }
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'message': f"Risk validation error: {e}"
            }
    
    async def _validate_drawdown_limits(self) -> Dict[str, Any]:
        """Validate drawdown limits"""
        try:
            current_drawdown = abs(min(self.daily_pnl, 0)) / self.portfolio_value
            
            return {
                'passed': current_drawdown <= self.max_drawdown,
                'current_drawdown': current_drawdown,
                'limit': self.max_drawdown,
                'daily_pnl': self.daily_pnl
            }
        except Exception as e:
            return {
                'passed': False,
                'error': str(e)
            }
    
    async def _validate_leverage_limits(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate leverage limits"""
        try:
            # Calculate current leverage
            total_position_value = sum(pos.position_value for pos in self.positions.values())
            
            # Add proposed trade
            new_trade_value = trade_data.get('size', 0) * self.cl_contract_size * trade_data.get('price', 0)
            new_total_value = total_position_value + new_trade_value
            
            current_leverage = new_total_value / self.portfolio_value
            
            return {
                'passed': current_leverage <= self.leverage_limit,
                'current_leverage': current_leverage,
                'limit': self.leverage_limit,
                'position_value': new_total_value
            }
        except Exception as e:
            return {
                'passed': False,
                'error': str(e)
            }
    
    async def _validate_market_conditions(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate market conditions for trading"""
        try:
            # Check market hours and liquidity
            current_hour = datetime.now().hour
            
            # ETH trading allowed but with reduced size
            if 0 <= current_hour <= 4 or 18 <= current_hour <= 23:
                return {
                    'passed': True,
                    'message': "ETH trading session - reduced liquidity",
                    'adjustment': 0.8  # Reduce position size by 20%
                }
            else:
                return {
                    'passed': True,
                    'message': "Regular trading hours - normal liquidity"
                }
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'message': f"Market conditions validation error: {e}"
            }
    
    def _record_risk_breach(self, trade_data: Dict[str, Any], validation_result: Dict[str, Any]):
        """Record risk limit breach"""
        breach = {
            'timestamp': datetime.now().isoformat(),
            'trade_data': trade_data,
            'validation_result': validation_result,
            'breach_type': 'trade_validation',
            'portfolio_value': self.portfolio_value,
            'current_drawdown': abs(min(self.daily_pnl, 0)) / self.portfolio_value
        }
        self.risk_breaches.append(breach)
        
        # Create alert
        alert = CLRiskAlert(
            alert_id=f"breach_{int(time.time())}",
            level=RiskLevel.HIGH,
            message=f"Risk limit breach: {trade_data.get('symbol', 'UNKNOWN')}",
            timestamp=datetime.now(),
            metrics=breach,
            action_required=True
        )
        self.alerts.append(alert)
    
    async def monitor_positions(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Monitor existing positions and generate alerts
        
        Args:
            market_data: Current market data
            
        Returns:
            Monitoring report
        """
        monitoring_report = {
            'timestamp': datetime.now().isoformat(),
            'positions_monitored': len(self.positions),
            'alerts_generated': 0,
            'risk_metrics': {},
            'portfolio_health': {}
        }
        
        try:
            # Update position values
            await self._update_position_values(market_data)
            
            # Calculate portfolio metrics
            portfolio_metrics = await self._calculate_portfolio_metrics()
            monitoring_report['portfolio_health'] = portfolio_metrics
            
            # Check for risk alerts
            alerts = await self._check_risk_alerts(market_data)
            monitoring_report['alerts_generated'] = len(alerts)
            
            # Update risk metrics
            risk_metrics = await self._calculate_cl_risk_metrics(market_data)
            monitoring_report['risk_metrics'] = risk_metrics.__dict__
            
            # Store metrics for history
            self.risk_metrics_history.append({
                'timestamp': datetime.now().isoformat(),
                'metrics': risk_metrics.__dict__,
                'portfolio_value': self.portfolio_value,
                'positions': len(self.positions)
            })
            
        except Exception as e:
            logger.error(f"Error monitoring positions: {e}")
            monitoring_report['error'] = str(e)
        
        return monitoring_report
    
    async def _update_position_values(self, market_data: Dict[str, Any]):
        """Update position values with current market data"""
        current_price = market_data.get('close', 0)
        
        for position in self.positions.values():
            position.current_price = current_price
            position.position_value = position.quantity * self.cl_contract_size * current_price
            
            # Calculate unrealized P&L
            if position.side == 'long':
                position.unrealized_pnl = (current_price - position.entry_price) * position.quantity * self.cl_contract_size
            else:
                position.unrealized_pnl = (position.entry_price - current_price) * position.quantity * self.cl_contract_size
    
    async def _calculate_portfolio_metrics(self) -> Dict[str, Any]:
        """Calculate portfolio health metrics"""
        try:
            # Calculate total exposure
            total_long_exposure = sum(pos.position_value for pos in self.positions.values() if pos.side == 'long')
            total_short_exposure = sum(pos.position_value for pos in self.positions.values() if pos.side == 'short')
            gross_exposure = total_long_exposure + total_short_exposure
            net_exposure = total_long_exposure - total_short_exposure
            
            # Calculate P&L
            total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
            total_realized_pnl = sum(pos.realized_pnl for pos in self.positions.values())
            
            # Calculate leverage
            leverage = gross_exposure / self.portfolio_value if self.portfolio_value > 0 else 0
            
            # Calculate drawdown
            peak_value = max(self.portfolio_value + total_unrealized_pnl, self.portfolio_value)
            current_value = self.portfolio_value + total_unrealized_pnl
            drawdown = (peak_value - current_value) / peak_value if peak_value > 0 else 0
            
            return {
                'portfolio_value': self.portfolio_value,
                'total_exposure': gross_exposure,
                'net_exposure': net_exposure,
                'leverage': leverage,
                'unrealized_pnl': total_unrealized_pnl,
                'realized_pnl': total_realized_pnl,
                'drawdown': drawdown,
                'num_positions': len(self.positions),
                'risk_utilization': min(gross_exposure / (self.portfolio_value * self.max_position_size), 1.0)
            }
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {e}")
            return {}
    
    async def _check_risk_alerts(self, market_data: Dict[str, Any]) -> List[CLRiskAlert]:
        """Check for risk alerts"""
        alerts = []
        
        try:
            # Check drawdown alerts
            portfolio_metrics = await self._calculate_portfolio_metrics()
            drawdown = portfolio_metrics.get('drawdown', 0)
            
            if drawdown > self.max_drawdown * 0.8:  # 80% of max drawdown
                alert = CLRiskAlert(
                    alert_id=f"drawdown_{int(time.time())}",
                    level=RiskLevel.HIGH if drawdown > self.max_drawdown * 0.9 else RiskLevel.MEDIUM,
                    message=f"High drawdown detected: {drawdown:.2%}",
                    timestamp=datetime.now(),
                    metrics={'drawdown': drawdown, 'limit': self.max_drawdown},
                    action_required=drawdown > self.max_drawdown * 0.9
                )
                alerts.append(alert)
            
            # Check leverage alerts
            leverage = portfolio_metrics.get('leverage', 0)
            if leverage > self.leverage_limit * 0.8:  # 80% of max leverage
                alert = CLRiskAlert(
                    alert_id=f"leverage_{int(time.time())}",
                    level=RiskLevel.HIGH if leverage > self.leverage_limit * 0.9 else RiskLevel.MEDIUM,
                    message=f"High leverage detected: {leverage:.2f}x",
                    timestamp=datetime.now(),
                    metrics={'leverage': leverage, 'limit': self.leverage_limit},
                    action_required=leverage > self.leverage_limit * 0.9
                )
                alerts.append(alert)
            
            # Check position-specific alerts
            for position in self.positions.values():
                # Check stop loss triggers
                if position.stop_loss:
                    current_price = market_data.get('close', 0)
                    if ((position.side == 'long' and current_price <= position.stop_loss) or
                        (position.side == 'short' and current_price >= position.stop_loss)):
                        
                        alert = CLRiskAlert(
                            alert_id=f"stop_loss_{position.symbol}_{int(time.time())}",
                            level=RiskLevel.CRITICAL,
                            message=f"Stop loss triggered: {position.symbol}",
                            timestamp=datetime.now(),
                            metrics={'position': position.__dict__, 'current_price': current_price},
                            action_required=True
                        )
                        alerts.append(alert)
            
            # Store alerts
            self.alerts.extend(alerts)
            
        except Exception as e:
            logger.error(f"Error checking risk alerts: {e}")
        
        return alerts
    
    def generate_risk_report(self) -> Dict[str, Any]:
        """Generate comprehensive risk report"""
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'portfolio_summary': {
                'portfolio_value': self.portfolio_value,
                'total_positions': len(self.positions),
                'daily_pnl': self.daily_pnl,
                'total_pnl': self.total_pnl,
                'max_drawdown_reached': self.max_drawdown_reached
            },
            'risk_limits': {
                'max_risk_per_trade': self.max_risk_per_trade,
                'max_position_size': self.max_position_size,
                'max_drawdown': self.max_drawdown,
                'daily_loss_limit': self.daily_loss_limit,
                'leverage_limit': self.leverage_limit
            },
            'performance_metrics': {
                'trades_executed': self.trades_executed,
                'trades_stopped': self.trades_stopped,
                'win_rate': self.trades_executed / max(self.trades_executed + self.trades_stopped, 1),
                'risk_adjusted_returns': self.risk_adjusted_returns[-20:] if self.risk_adjusted_returns else []
            },
            'alerts_summary': {
                'total_alerts': len(self.alerts),
                'critical_alerts': len([a for a in self.alerts if a.level == RiskLevel.CRITICAL]),
                'high_alerts': len([a for a in self.alerts if a.level == RiskLevel.HIGH]),
                'recent_alerts': [a.__dict__ for a in list(self.alerts)[-5:]]
            },
            'risk_breaches': {
                'total_breaches': len(self.risk_breaches),
                'recent_breaches': list(self.risk_breaches)[-5:] if self.risk_breaches else []
            },
            'market_conditions': {
                'current_session': 'ETH' if (0 <= datetime.now().hour <= 4 or 18 <= datetime.now().hour <= 23) else 'RTH',
                'liquidity_score': self._calculate_session_liquidity({}),
                'volatility_regime': 'HIGH' if len(self.risk_metrics_history) > 0 and 
                                   self.risk_metrics_history[-1]['metrics'].get('atr_20', 0) > 1.0 else 'NORMAL'
            },
            'recommendations': self._generate_risk_recommendations()
        }
        
        return report
    
    def _generate_risk_recommendations(self) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []
        
        # Analyze recent performance
        if len(self.risk_breaches) > 5:
            recommendations.append("Consider reducing position sizes - multiple risk breaches detected")
        
        if self.trades_stopped > self.trades_executed * 0.3:
            recommendations.append("High stop-out rate - review stop loss levels")
        
        # Check portfolio concentration
        if len(self.positions) > 0:
            cl_exposure = sum(pos.position_value for pos in self.positions.values())
            if cl_exposure > self.portfolio_value * 0.6:
                recommendations.append("High CL concentration - consider diversification")
        
        # Check recent alerts
        critical_alerts = [a for a in self.alerts if a.level == RiskLevel.CRITICAL]
        if len(critical_alerts) > 0:
            recommendations.append("Critical alerts active - immediate attention required")
        
        # Market condition recommendations
        current_hour = datetime.now().hour
        if 0 <= current_hour <= 4 or 18 <= current_hour <= 23:
            recommendations.append("ETH trading session - consider reduced position sizes")
        
        if not recommendations:
            recommendations.append("Risk management operating within acceptable parameters")
        
        return recommendations
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get real-time dashboard data"""
        try:
            # Calculate key metrics
            portfolio_metrics = asyncio.run(self._calculate_portfolio_metrics())
            
            # Get recent alerts
            recent_alerts = [a.__dict__ for a in list(self.alerts)[-5:]]
            
            # Calculate risk utilization
            risk_utilization = {
                'position_size': portfolio_metrics.get('risk_utilization', 0),
                'leverage': portfolio_metrics.get('leverage', 0) / self.leverage_limit,
                'drawdown': portfolio_metrics.get('drawdown', 0) / self.max_drawdown
            }
            
            return {
                'timestamp': datetime.now().isoformat(),
                'portfolio_value': self.portfolio_value,
                'daily_pnl': self.daily_pnl,
                'total_pnl': self.total_pnl,
                'positions': len(self.positions),
                'active_alerts': len([a for a in self.alerts if a.action_required]),
                'risk_utilization': risk_utilization,
                'recent_alerts': recent_alerts,
                'performance_metrics': {
                    'trades_executed': self.trades_executed,
                    'trades_stopped': self.trades_stopped,
                    'win_rate': self.trades_executed / max(self.trades_executed + self.trades_stopped, 1) if self.trades_executed + self.trades_stopped > 0 else 0
                },
                'market_status': {
                    'session': 'ETH' if (0 <= datetime.now().hour <= 4 or 18 <= datetime.now().hour <= 23) else 'RTH',
                    'liquidity': self._calculate_session_liquidity({})
                }
            }
        except Exception as e:
            logger.error(f"Error getting dashboard data: {e}")
            return {'error': str(e)}
    
    def reset_daily_metrics(self):
        """Reset daily metrics at market open"""
        self.daily_pnl = 0.0
        
        # Clear daily alerts
        self.alerts = deque([a for a in self.alerts if a.level == RiskLevel.CRITICAL], maxlen=100)
        
        logger.info("Daily risk metrics reset")