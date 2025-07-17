"""
Realistic Execution Engine for NQ Futures Trading
================================================

AGENT 3 - TRADING EXECUTION REALISM SPECIALIST

This module implements realistic execution conditions for NQ futures trading:
- Realistic slippage based on market conditions and position size
- Proper NQ futures commission structure ($0.50 per round turn)
- Execution latency simulation (signal to order to fill delays)
- Market impact modeling for larger positions
- Bid-ask spread simulation
- Proper margin requirements and position sizing

NQ Futures Specifications:
- Contract value: $20 per point (E-mini NASDAQ-100)
- Tick size: 0.25 points = $5.00
- Commission: $0.50 per round turn
- Initial margin: ~$19,000 per contract
- Maintenance margin: ~$17,300 per contract

Author: AGENT 3 - Trading Execution Realism Specialist
Date: 2025-07-16
Mission: Transform perfect execution into realistic trading conditions
"""

import numpy as np
import pandas as pd
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types for execution"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_MARKET = "stop_market"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order side"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order execution status"""
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL_FILL = "partial_fill"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class NQFuturesSpecs:
    """NQ E-mini futures contract specifications"""
    contract_name: str = "NQ"
    point_value: float = 20.0  # $20 per point
    tick_size: float = 0.25    # 0.25 points
    tick_value: float = 5.0    # $5 per tick
    commission_per_rt: float = 0.50  # $0.50 per round turn
    initial_margin: float = 19000.0  # $19,000 per contract
    maintenance_margin: float = 17300.0  # $17,300 per contract
    exchange_fees: float = 0.02  # $0.02 exchange fees
    nfa_fees: float = 0.02       # $0.02 NFA fees


@dataclass
class MarketConditions:
    """Current market conditions affecting execution"""
    timestamp: datetime
    current_price: float
    bid_price: float
    ask_price: float
    bid_size: int
    ask_size: int
    volume_rate: float  # Current volume vs average
    volatility_regime: float  # 0.0 = low vol, 1.0 = high vol
    time_of_day_factor: float  # 0.0 = illiquid hours, 1.0 = peak hours
    stress_indicator: float  # 0.0 = calm, 1.0 = stressed


@dataclass
class ExecutionOrder:
    """Order for execution"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int  # Number of contracts
    price: Optional[float] = None  # For limit orders
    stop_price: Optional[float] = None  # For stop orders
    timestamp_created: datetime = field(default_factory=datetime.now)
    
    # Position sizing parameters
    risk_amount: float = 0.0  # Risk amount in dollars
    account_value: float = 0.0
    risk_percent: float = 0.0  # Risk as % of account
    
    # Execution results (filled after execution)
    fill_price: Optional[float] = None
    fill_quantity: int = 0
    fill_timestamp: Optional[datetime] = None
    status: OrderStatus = OrderStatus.PENDING
    slippage_points: float = 0.0
    commission_paid: float = 0.0
    total_fees: float = 0.0
    latency_ms: float = 0.0


@dataclass
class ExecutionResult:
    """Result of order execution"""
    order: ExecutionOrder
    execution_success: bool
    fill_details: Dict[str, Any]
    market_impact: float  # Price impact in points
    timing_breakdown: Dict[str, float]  # Latency breakdown
    cost_breakdown: Dict[str, float]  # Cost breakdown
    reasoning: str


class RealisticSlippageModel:
    """
    Realistic slippage model for NQ futures based on:
    - Position size (market impact)
    - Market conditions (volatility, liquidity)
    - Time of day
    - Order type
    """
    
    def __init__(self, specs: NQFuturesSpecs):
        self.specs = specs
        
        # Base slippage parameters (in points)
        self.base_slippage_market = 0.5  # 0.5 points base for market orders
        self.base_slippage_limit = 0.25  # 0.25 points base for limit orders
        
        # Market impact factors
        self.impact_per_contract = 0.1  # 0.1 points per additional contract
        self.volatility_multiplier = 2.0  # Multiply slippage by volatility
        self.liquidity_factor = 1.5  # Multiply slippage when illiquid
        
    def calculate_slippage(self, order: ExecutionOrder, 
                          market_conditions: MarketConditions) -> float:
        """
        Calculate realistic slippage in points
        
        Returns:
            Slippage in points (always positive, represents cost)
        """
        # Base slippage by order type
        if order.order_type == OrderType.MARKET:
            base_slippage = self.base_slippage_market
        else:
            base_slippage = self.base_slippage_limit
            
        # Market impact based on position size
        market_impact = order.quantity * self.impact_per_contract
        
        # Volatility adjustment
        volatility_adjustment = market_conditions.volatility_regime * self.volatility_multiplier
        
        # Liquidity adjustment (worse during off-hours)
        liquidity_adjustment = (1.0 - market_conditions.time_of_day_factor) * self.liquidity_factor
        
        # Volume impact (high volume = better execution)
        volume_adjustment = max(0.5, 1.0 / market_conditions.volume_rate)
        
        # Stress adjustment (market stress increases slippage)
        stress_adjustment = 1.0 + market_conditions.stress_indicator
        
        # Calculate total slippage
        total_slippage = (
            base_slippage + 
            market_impact + 
            volatility_adjustment + 
            liquidity_adjustment
        ) * volume_adjustment * stress_adjustment
        
        # Ensure minimum slippage (at least 1 tick)
        total_slippage = max(total_slippage, self.specs.tick_size)
        
        return total_slippage


class ExecutionLatencyModel:
    """
    Models realistic execution latency for NQ futures
    """
    
    def __init__(self):
        # Latency components in milliseconds
        self.signal_processing_ms = (50, 150)  # Signal generation to decision
        self.order_routing_ms = (20, 80)       # Order routing to exchange
        self.exchange_processing_ms = (5, 30)  # Exchange processing time
        self.fill_confirmation_ms = (10, 50)   # Fill confirmation back
        
        # Market condition adjustments
        self.stress_multiplier = 2.0  # Multiply latency during stress
        self.volume_divisor = 1.5     # Divide latency during high volume
        
    def calculate_execution_latency(self, 
                                  market_conditions: MarketConditions) -> Dict[str, float]:
        """
        Calculate realistic execution latency breakdown
        
        Returns:
            Dictionary with latency components in milliseconds
        """
        # Base latency components
        signal_processing = np.random.uniform(*self.signal_processing_ms)
        order_routing = np.random.uniform(*self.order_routing_ms)
        exchange_processing = np.random.uniform(*self.exchange_processing_ms)
        fill_confirmation = np.random.uniform(*self.fill_confirmation_ms)
        
        # Market condition adjustments
        stress_factor = 1.0 + market_conditions.stress_indicator * (self.stress_multiplier - 1.0)
        volume_factor = max(0.5, 1.0 / (market_conditions.volume_rate / self.volume_divisor))
        
        # Apply adjustments
        total_factor = stress_factor * volume_factor
        
        latency_breakdown = {
            'signal_processing_ms': signal_processing * total_factor,
            'order_routing_ms': order_routing * total_factor,
            'exchange_processing_ms': exchange_processing * total_factor,
            'fill_confirmation_ms': fill_confirmation * total_factor
        }
        
        latency_breakdown['total_latency_ms'] = sum(latency_breakdown.values())
        
        return latency_breakdown


class PositionSizingFramework:
    """
    Risk-based position sizing for NQ futures
    """
    
    def __init__(self, specs: NQFuturesSpecs):
        self.specs = specs
        
        # Default risk parameters
        self.default_risk_percent = 0.02  # 2% of account per trade
        self.max_risk_percent = 0.05      # 5% maximum risk
        self.min_contracts = 1            # Minimum 1 contract
        self.max_contracts_per_trade = 10 # Maximum 10 contracts per trade
        
    def calculate_position_size(self, 
                               account_value: float,
                               entry_price: float,
                               stop_loss_price: float,
                               risk_percent: Optional[float] = None) -> Dict[str, Any]:
        """
        Calculate optimal position size based on risk parameters
        
        Args:
            account_value: Total account value
            entry_price: Planned entry price
            stop_loss_price: Stop loss price
            risk_percent: Risk percentage (default: 2%)
            
        Returns:
            Position sizing analysis
        """
        if risk_percent is None:
            risk_percent = self.default_risk_percent
            
        # Validate risk percent
        risk_percent = min(risk_percent, self.max_risk_percent)
        
        # Calculate risk amount in dollars
        risk_amount = account_value * risk_percent
        
        # Calculate risk per contract in points
        risk_per_contract_points = abs(entry_price - stop_loss_price)
        
        # Calculate risk per contract in dollars
        risk_per_contract_dollars = risk_per_contract_points * self.specs.point_value
        
        # Calculate optimal number of contracts
        if risk_per_contract_dollars > 0:
            optimal_contracts = risk_amount / risk_per_contract_dollars
        else:
            optimal_contracts = self.min_contracts
            
        # Apply constraints
        contracts = max(self.min_contracts, min(int(optimal_contracts), self.max_contracts_per_trade))
        
        # Calculate actual risk
        actual_risk_dollars = contracts * risk_per_contract_dollars
        actual_risk_percent = actual_risk_dollars / account_value
        
        # Margin requirement check
        required_margin = contracts * self.specs.initial_margin
        margin_available = account_value * 0.8  # Use 80% of account for margin
        
        # Adjust for margin if needed
        if required_margin > margin_available:
            max_contracts_by_margin = int(margin_available / self.specs.initial_margin)
            contracts = min(contracts, max_contracts_by_margin)
            
            # Recalculate actual risk
            actual_risk_dollars = contracts * risk_per_contract_dollars
            actual_risk_percent = actual_risk_dollars / account_value
        
        return {
            'contracts': contracts,
            'risk_amount_target': risk_amount,
            'risk_amount_actual': actual_risk_dollars,
            'risk_percent_target': risk_percent,
            'risk_percent_actual': actual_risk_percent,
            'risk_per_contract_points': risk_per_contract_points,
            'risk_per_contract_dollars': risk_per_contract_dollars,
            'required_margin': contracts * self.specs.initial_margin,
            'margin_utilization': (contracts * self.specs.initial_margin) / account_value,
            'position_value': contracts * entry_price * self.specs.point_value
        }


class RealisticPnLCalculator:
    """
    Realistic PnL calculation including all transaction costs
    """
    
    def __init__(self, specs: NQFuturesSpecs):
        self.specs = specs
        
    def calculate_trade_pnl(self, 
                           entry_order: ExecutionOrder,
                           exit_order: ExecutionOrder) -> Dict[str, float]:
        """
        Calculate complete trade PnL including all costs
        
        Returns:
            Complete PnL breakdown
        """
        # Basic PnL calculation
        if entry_order.side == OrderSide.BUY:
            price_diff_points = exit_order.fill_price - entry_order.fill_price
        else:
            price_diff_points = entry_order.fill_price - exit_order.fill_price
            
        # Gross PnL (before costs)
        gross_pnl = price_diff_points * entry_order.quantity * self.specs.point_value
        
        # Calculate all costs
        entry_commission = entry_order.commission_paid
        exit_commission = exit_order.commission_paid
        total_commission = entry_commission + exit_commission
        
        entry_fees = entry_order.total_fees - entry_order.commission_paid
        exit_fees = exit_order.total_fees - exit_order.commission_paid
        total_fees = entry_fees + exit_fees
        
        # Slippage costs
        entry_slippage_cost = entry_order.slippage_points * entry_order.quantity * self.specs.point_value
        exit_slippage_cost = exit_order.slippage_points * exit_order.quantity * self.specs.point_value
        total_slippage_cost = entry_slippage_cost + exit_slippage_cost
        
        # Net PnL
        net_pnl = gross_pnl - total_commission - total_fees - total_slippage_cost
        
        # Additional metrics
        total_costs = total_commission + total_fees + total_slippage_cost
        cost_per_contract = total_costs / entry_order.quantity if entry_order.quantity > 0 else 0
        
        return {
            'gross_pnl': gross_pnl,
            'net_pnl': net_pnl,
            'price_diff_points': price_diff_points,
            'total_commission': total_commission,
            'total_fees': total_fees,
            'total_slippage_cost': total_slippage_cost,
            'total_costs': total_costs,
            'cost_per_contract': cost_per_contract,
            'cost_percentage': (total_costs / abs(gross_pnl)) * 100 if gross_pnl != 0 else 0,
            'return_on_margin': net_pnl / (entry_order.quantity * self.specs.initial_margin) * 100
        }
        
    def calculate_unrealized_pnl(self,
                                entry_order: ExecutionOrder,
                                current_price: float) -> Dict[str, float]:
        """
        Calculate unrealized PnL for open position
        """
        if entry_order.side == OrderSide.BUY:
            price_diff_points = current_price - entry_order.fill_price
        else:
            price_diff_points = entry_order.fill_price - current_price
            
        unrealized_pnl = price_diff_points * entry_order.quantity * self.specs.point_value
        
        # Include entry costs but not exit costs (not yet realized)
        entry_costs = entry_order.commission_paid + entry_order.total_fees + \
                     (entry_order.slippage_points * entry_order.quantity * self.specs.point_value)
        
        net_unrealized_pnl = unrealized_pnl - entry_costs
        
        return {
            'unrealized_pnl_gross': unrealized_pnl,
            'unrealized_pnl_net': net_unrealized_pnl,
            'price_diff_points': price_diff_points,
            'entry_costs': entry_costs,
            'return_on_margin': net_unrealized_pnl / (entry_order.quantity * self.specs.initial_margin) * 100
        }


class RealisticExecutionEngine:
    """
    Main realistic execution engine for NQ futures
    """
    
    def __init__(self, 
                 account_value: float = 100000.0,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize realistic execution engine
        
        Args:
            account_value: Starting account value
            config: Configuration parameters
        """
        self.account_value = account_value
        self.config = config or {}
        
        # Initialize components
        self.specs = NQFuturesSpecs()
        self.slippage_model = RealisticSlippageModel(self.specs)
        self.latency_model = ExecutionLatencyModel()
        self.position_sizer = PositionSizingFramework(self.specs)
        self.pnl_calculator = RealisticPnLCalculator(self.specs)
        
        # Execution tracking
        self.executed_orders: List[ExecutionOrder] = []
        self.open_positions: List[ExecutionOrder] = []
        self.closed_trades: List[Tuple[ExecutionOrder, ExecutionOrder]] = []
        
        # Performance metrics
        self.total_commission_paid = 0.0
        self.total_slippage_cost = 0.0
        self.total_trades = 0
        self.successful_fills = 0
        self.failed_fills = 0
        
        logger.info(f"RealisticExecutionEngine initialized with ${account_value:,.2f} account value")
        
    def create_market_conditions(self, 
                                current_price: float,
                                timestamp: datetime,
                                volume_data: Optional[Dict] = None) -> MarketConditions:
        """
        Create market conditions based on current market data
        """
        # Simple bid-ask spread simulation (typically 0.25-1.0 points for NQ)
        spread_points = np.random.uniform(0.25, 1.0)
        bid_price = current_price - spread_points / 2
        ask_price = current_price + spread_points / 2
        
        # Simulate market depth
        bid_size = np.random.randint(10, 100)
        ask_size = np.random.randint(10, 100)
        
        # Time of day factor (market hours vs off-hours)
        hour = timestamp.hour
        if 9 <= hour <= 16:  # Market hours
            time_of_day_factor = 1.0
        elif 6 <= hour <= 9 or 16 <= hour <= 18:  # Pre/post market
            time_of_day_factor = 0.7
        else:  # Overnight
            time_of_day_factor = 0.3
            
        # Volume rate (simplified)
        volume_rate = volume_data.get('volume_ratio', 1.0) if volume_data else 1.0
        
        # Volatility regime (simplified)
        volatility_regime = min(1.0, volume_data.get('volatility', 0.5)) if volume_data else 0.5
        
        # Stress indicator (simplified)
        stress_indicator = volume_data.get('stress', 0.0) if volume_data else 0.0
        
        return MarketConditions(
            timestamp=timestamp,
            current_price=current_price,
            bid_price=bid_price,
            ask_price=ask_price,
            bid_size=bid_size,
            ask_size=ask_size,
            volume_rate=volume_rate,
            volatility_regime=volatility_regime,
            time_of_day_factor=time_of_day_factor,
            stress_indicator=stress_indicator
        )
        
    async def execute_order(self, 
                           order: ExecutionOrder,
                           market_conditions: MarketConditions) -> ExecutionResult:
        """
        Execute order with realistic conditions
        
        Returns:
            Detailed execution result
        """
        start_time = time.perf_counter()
        
        try:
            # 1. Calculate execution latency
            latency_breakdown = self.latency_model.calculate_execution_latency(market_conditions)
            
            # Simulate latency delay
            total_latency_seconds = latency_breakdown['total_latency_ms'] / 1000.0
            await asyncio.sleep(min(total_latency_seconds, 0.5))  # Cap at 500ms for simulation
            
            # 2. Calculate slippage
            slippage_points = self.slippage_model.calculate_slippage(order, market_conditions)
            
            # 3. Determine fill price based on order type and slippage
            fill_price = self._calculate_fill_price(order, market_conditions, slippage_points)
            
            # 4. Calculate costs
            commission = order.quantity * self.specs.commission_per_rt
            exchange_fees = order.quantity * self.specs.exchange_fees
            nfa_fees = order.quantity * self.specs.nfa_fees
            total_fees = commission + exchange_fees + nfa_fees
            
            # 5. Market impact simulation (price moves after large orders)
            market_impact = self._calculate_market_impact(order, market_conditions)
            
            # 6. Update order with execution details
            order.fill_price = fill_price
            order.fill_quantity = order.quantity  # Assume full fill for simplicity
            order.fill_timestamp = datetime.now()
            order.status = OrderStatus.FILLED
            order.slippage_points = slippage_points
            order.commission_paid = commission
            order.total_fees = total_fees
            order.latency_ms = latency_breakdown['total_latency_ms']
            
            # 7. Update tracking
            self.executed_orders.append(order)
            self.total_commission_paid += commission
            self.total_slippage_cost += slippage_points * order.quantity * self.specs.point_value
            self.total_trades += 1
            self.successful_fills += 1
            
            # 8. Update positions
            if order.side == OrderSide.BUY:
                self.open_positions.append(order)
            else:
                # Close position (simplified - assume FIFO)
                if self.open_positions:
                    entry_order = self.open_positions.pop(0)
                    self.closed_trades.append((entry_order, order))
            
            # 9. Create execution result
            execution_result = ExecutionResult(
                order=order,
                execution_success=True,
                fill_details={
                    'fill_price': fill_price,
                    'fill_quantity': order.quantity,
                    'slippage_points': slippage_points,
                    'slippage_cost': slippage_points * order.quantity * self.specs.point_value
                },
                market_impact=market_impact,
                timing_breakdown=latency_breakdown,
                cost_breakdown={
                    'commission': commission,
                    'exchange_fees': exchange_fees,
                    'nfa_fees': nfa_fees,
                    'total_fees': total_fees,
                    'slippage_cost': slippage_points * order.quantity * self.specs.point_value
                },
                reasoning=f"Market order executed with {slippage_points:.2f} points slippage"
            )
            
            logger.info(f"Order executed: {order.order_id} - {order.quantity} contracts @ {fill_price:.2f}")
            
            return execution_result
            
        except Exception as e:
            # Handle execution failure
            order.status = OrderStatus.REJECTED
            self.failed_fills += 1
            
            logger.error(f"Order execution failed: {order.order_id} - {str(e)}")
            
            return ExecutionResult(
                order=order,
                execution_success=False,
                fill_details={},
                market_impact=0.0,
                timing_breakdown={},
                cost_breakdown={},
                reasoning=f"Execution failed: {str(e)}"
            )
            
    def _calculate_fill_price(self, 
                             order: ExecutionOrder,
                             market_conditions: MarketConditions,
                             slippage_points: float) -> float:
        """
        Calculate realistic fill price including slippage
        """
        if order.order_type == OrderType.MARKET:
            if order.side == OrderSide.BUY:
                # Market buy: start with ask price + slippage
                fill_price = market_conditions.ask_price + slippage_points
            else:
                # Market sell: start with bid price - slippage
                fill_price = market_conditions.bid_price - slippage_points
        else:
            # For limit orders, use order price (assume immediate fill for simplicity)
            fill_price = order.price
            
        return round(fill_price / self.specs.tick_size) * self.specs.tick_size
        
    def _calculate_market_impact(self, 
                                order: ExecutionOrder,
                                market_conditions: MarketConditions) -> float:
        """
        Calculate market impact from order execution
        """
        # Simple market impact model: larger orders move price more
        base_impact = order.quantity * 0.05  # 0.05 points per contract
        
        # Adjust for market conditions
        liquidity_factor = 1.0 / market_conditions.time_of_day_factor
        volume_factor = 1.0 / market_conditions.volume_rate
        
        market_impact = base_impact * liquidity_factor * volume_factor
        
        return min(market_impact, 2.0)  # Cap at 2 points
        
    def create_order(self,
                    side: OrderSide,
                    quantity: int,
                    order_type: OrderType = OrderType.MARKET,
                    price: Optional[float] = None,
                    stop_price: Optional[float] = None,
                    risk_percent: Optional[float] = None) -> ExecutionOrder:
        """
        Create an order with proper position sizing
        """
        order_id = f"NQ_{int(time.time() * 1000)}"
        
        # Calculate position size if not specified
        if risk_percent and stop_price and price:
            sizing_result = self.position_sizer.calculate_position_size(
                self.account_value, price, stop_price, risk_percent
            )
            quantity = sizing_result['contracts']
        
        order = ExecutionOrder(
            order_id=order_id,
            symbol="NQ",
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            risk_amount=quantity * abs(price - stop_price) * self.specs.point_value if price and stop_price else 0.0,
            account_value=self.account_value,
            risk_percent=risk_percent or 0.02
        )
        
        return order
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive performance metrics
        """
        total_orders = len(self.executed_orders)
        fill_rate = self.successful_fills / total_orders if total_orders > 0 else 0.0
        
        # Calculate average slippage
        total_slippage_points = sum(order.slippage_points for order in self.executed_orders)
        avg_slippage_points = total_slippage_points / total_orders if total_orders > 0 else 0.0
        
        # Calculate average latency
        total_latency = sum(order.latency_ms for order in self.executed_orders)
        avg_latency_ms = total_latency / total_orders if total_orders > 0 else 0.0
        
        # Calculate total costs
        avg_commission_per_trade = self.total_commission_paid / total_orders if total_orders > 0 else 0.0
        
        # PnL analysis for closed trades
        total_net_pnl = 0.0
        winning_trades = 0
        losing_trades = 0
        
        for entry_order, exit_order in self.closed_trades:
            trade_pnl = self.pnl_calculator.calculate_trade_pnl(entry_order, exit_order)
            total_net_pnl += trade_pnl['net_pnl']
            
            if trade_pnl['net_pnl'] > 0:
                winning_trades += 1
            else:
                losing_trades += 1
        
        win_rate = winning_trades / len(self.closed_trades) if self.closed_trades else 0.0
        
        return {
            'execution_metrics': {
                'total_orders': total_orders,
                'successful_fills': self.successful_fills,
                'failed_fills': self.failed_fills,
                'fill_rate': fill_rate,
                'avg_slippage_points': avg_slippage_points,
                'avg_slippage_cost': self.total_slippage_cost / total_orders if total_orders > 0 else 0.0,
                'avg_latency_ms': avg_latency_ms,
                'avg_commission_per_trade': avg_commission_per_trade
            },
            'trading_metrics': {
                'total_trades_closed': len(self.closed_trades),
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'total_net_pnl': total_net_pnl,
                'total_commission_paid': self.total_commission_paid,
                'total_slippage_cost': self.total_slippage_cost
            },
            'position_metrics': {
                'open_positions': len(self.open_positions),
                'total_margin_used': sum(pos.quantity * self.specs.initial_margin for pos in self.open_positions),
                'margin_utilization': sum(pos.quantity * self.specs.initial_margin for pos in self.open_positions) / self.account_value
            }
        }
        
    def save_execution_report(self, file_path: Optional[str] = None) -> str:
        """
        Save detailed execution report
        """
        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"results/execution_report_{timestamp}.json"
            
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        report = {
            'execution_engine_config': {
                'account_value': self.account_value,
                'nq_specs': {
                    'point_value': self.specs.point_value,
                    'tick_size': self.specs.tick_size,
                    'commission_per_rt': self.specs.commission_per_rt,
                    'initial_margin': self.specs.initial_margin
                }
            },
            'performance_metrics': self.get_performance_metrics(),
            'executed_orders': [
                {
                    'order_id': order.order_id,
                    'side': order.side.value,
                    'quantity': order.quantity,
                    'fill_price': order.fill_price,
                    'slippage_points': order.slippage_points,
                    'commission_paid': order.commission_paid,
                    'latency_ms': order.latency_ms,
                    'timestamp': order.fill_timestamp.isoformat() if order.fill_timestamp else None
                }
                for order in self.executed_orders
            ],
            'closed_trades_pnl': [
                self.pnl_calculator.calculate_trade_pnl(entry, exit)
                for entry, exit in self.closed_trades
            ]
        }
        
        with open(file_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Execution report saved: {file_path}")
        return file_path


# Example usage and testing functions
async def test_realistic_execution():
    """
    Test the realistic execution engine
    """
    print("🧪 Testing Realistic Execution Engine for NQ Futures")
    print("=" * 60)
    
    # Initialize engine
    engine = RealisticExecutionEngine(account_value=100000.0)
    
    # Create test market conditions
    current_price = 15000.0
    timestamp = datetime.now()
    market_conditions = engine.create_market_conditions(
        current_price=current_price,
        timestamp=timestamp,
        volume_data={'volume_ratio': 1.2, 'volatility': 0.6, 'stress': 0.1}
    )
    
    print(f"Market Conditions:")
    print(f"  Current Price: {market_conditions.current_price:.2f}")
    print(f"  Bid/Ask: {market_conditions.bid_price:.2f}/{market_conditions.ask_price:.2f}")
    print(f"  Volume Rate: {market_conditions.volume_rate:.2f}")
    print(f"  Time Factor: {market_conditions.time_of_day_factor:.2f}")
    
    # Test position sizing
    sizing_result = engine.position_sizer.calculate_position_size(
        account_value=100000.0,
        entry_price=15000.0,
        stop_loss_price=14950.0,
        risk_percent=0.02
    )
    
    print(f"\nPosition Sizing (2% risk):")
    print(f"  Optimal Contracts: {sizing_result['contracts']}")
    print(f"  Risk Amount: ${sizing_result['risk_amount_actual']:,.2f}")
    print(f"  Margin Required: ${sizing_result['required_margin']:,.2f}")
    
    # Test order execution
    buy_order = engine.create_order(
        side=OrderSide.BUY,
        quantity=sizing_result['contracts'],
        order_type=OrderType.MARKET
    )
    
    print(f"\nExecuting Buy Order: {buy_order.quantity} contracts")
    execution_result = await engine.execute_order(buy_order, market_conditions)
    
    print(f"Execution Result:")
    print(f"  Success: {execution_result.execution_success}")
    print(f"  Fill Price: {execution_result.fill_details.get('fill_price', 'N/A'):.2f}")
    print(f"  Slippage: {execution_result.fill_details.get('slippage_points', 0):.2f} points")
    print(f"  Total Latency: {execution_result.timing_breakdown.get('total_latency_ms', 0):.1f}ms")
    print(f"  Commission: ${execution_result.cost_breakdown.get('commission', 0):.2f}")
    
    # Test sell order
    sell_order = engine.create_order(
        side=OrderSide.SELL,
        quantity=sizing_result['contracts'],
        order_type=OrderType.MARKET
    )
    
    # Simulate price movement
    market_conditions.current_price = 15025.0  # Price moved up
    market_conditions.bid_price = 15024.5
    market_conditions.ask_price = 15025.5
    
    print(f"\nExecuting Sell Order: {sell_order.quantity} contracts")
    execution_result = await engine.execute_order(sell_order, market_conditions)
    
    print(f"Execution Result:")
    print(f"  Success: {execution_result.execution_success}")
    print(f"  Fill Price: {execution_result.fill_details.get('fill_price', 'N/A'):.2f}")
    print(f"  Slippage: {execution_result.fill_details.get('slippage_points', 0):.2f} points")
    
    # Get performance metrics
    metrics = engine.get_performance_metrics()
    print(f"\nPerformance Metrics:")
    print(f"  Fill Rate: {metrics['execution_metrics']['fill_rate']:.1%}")
    print(f"  Avg Slippage: {metrics['execution_metrics']['avg_slippage_points']:.2f} points")
    print(f"  Avg Latency: {metrics['execution_metrics']['avg_latency_ms']:.1f}ms")
    print(f"  Total Commission: ${metrics['trading_metrics']['total_commission_paid']:.2f}")
    
    # Save report
    report_file = engine.save_execution_report()
    print(f"\nReport saved: {report_file}")
    
    return engine, metrics


if __name__ == "__main__":
    # Run test
    import asyncio
    asyncio.run(test_realistic_execution())