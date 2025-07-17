"""
Advanced Market Simulation Module

This module provides realistic market microstructure simulation for backtesting,
including bid-ask spreads, partial fills, market impact, and latency effects.

Features:
- Realistic bid-ask spread modeling
- Partial fill simulation with order book depth
- Market impact models (linear, square-root, power-law)
- Latency and slippage simulation
- Liquidity-based execution modeling
- Transaction cost analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from numba import jit
import asyncio
from collections import deque
import warnings

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types for market simulation"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order sides"""
    BUY = "buy"
    SELL = "sell"


@dataclass
class MarketData:
    """Market data snapshot"""
    timestamp: datetime
    symbol: str
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    last_price: float
    volume: float
    volatility: float
    

@dataclass
class OrderBookLevel:
    """Order book level data"""
    price: float
    size: float
    num_orders: int


@dataclass
class OrderBook:
    """Order book structure"""
    timestamp: datetime
    symbol: str
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    
    def best_bid(self) -> Optional[float]:
        """Get best bid price"""
        return self.bids[0].price if self.bids else None
    
    def best_ask(self) -> Optional[float]:
        """Get best ask price"""
        return self.asks[0].price if self.asks else None
    
    def spread(self) -> float:
        """Calculate bid-ask spread"""
        bid = self.best_bid()
        ask = self.best_ask()
        if bid and ask:
            return ask - bid
        return 0.0
    
    def mid_price(self) -> Optional[float]:
        """Calculate mid price"""
        bid = self.best_bid()
        ask = self.best_ask()
        if bid and ask:
            return (bid + ask) / 2
        return None


@dataclass
class Order:
    """Order representation"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    time_in_force: str = "DAY"
    
    # Execution tracking
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    status: str = "PENDING"
    fills: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Fill:
    """Fill representation"""
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime
    commission: float = 0.0
    market_impact: float = 0.0
    latency_ms: float = 0.0


@dataclass
class MarketImpactModel:
    """Market impact model parameters"""
    model_type: str = "square_root"  # "linear", "square_root", "power_law"
    temporary_impact: float = 0.001  # Basis points per sqrt(volume)
    permanent_impact: float = 0.0005  # Basis points per sqrt(volume)
    participation_rate: float = 0.1  # Fraction of volume
    decay_rate: float = 0.95  # Impact decay rate
    
    # Power law parameters
    alpha: float = 0.5  # Exponent for power law
    beta: float = 0.6   # Volume scaling


class MarketSimulator:
    """
    Advanced market simulator with realistic microstructure effects
    """
    
    def __init__(
        self,
        symbols: List[str],
        initial_prices: Dict[str, float],
        volatilities: Dict[str, float],
        spreads: Dict[str, float],
        market_impact_model: Optional[MarketImpactModel] = None,
        latency_model: Optional[Dict[str, Any]] = None
    ):
        self.symbols = symbols
        self.prices = initial_prices.copy()
        self.volatilities = volatilities
        self.spreads = spreads
        
        # Market impact model
        self.market_impact_model = market_impact_model or MarketImpactModel()
        
        # Latency model
        self.latency_model = latency_model or {
            'base_latency_ms': 1.0,
            'network_jitter_ms': 0.5,
            'processing_delay_ms': 0.1,
            'queue_delay_factor': 1.5
        }
        
        # Order book simulation
        self.order_books: Dict[str, OrderBook] = {}
        self.market_data: Dict[str, MarketData] = {}
        
        # Execution tracking
        self.pending_orders: Dict[str, Order] = {}
        self.filled_orders: Dict[str, Order] = {}
        self.fills: List[Fill] = []
        
        # Market state
        self.current_time = datetime.now()
        self.volume_history: Dict[str, deque] = {symbol: deque(maxlen=100) for symbol in symbols}
        self.price_history: Dict[str, deque] = {symbol: deque(maxlen=100) for symbol in symbols}
        
        # Performance tracking
        self.execution_stats = {
            'total_orders': 0,
            'total_fills': 0,
            'avg_fill_time_ms': 0.0,
            'avg_slippage_bps': 0.0,
            'avg_market_impact_bps': 0.0
        }
        
        # Initialize market data
        self._initialize_market_data()
    
    def _initialize_market_data(self):
        """Initialize market data and order books"""
        
        for symbol in self.symbols:
            price = self.prices[symbol]
            spread = self.spreads[symbol]
            
            # Create initial order book
            bid_price = price - spread / 2
            ask_price = price + spread / 2
            
            # Generate realistic order book depth
            bid_levels = []
            ask_levels = []
            
            for i in range(10):  # 10 levels deep
                bid_levels.append(OrderBookLevel(
                    price=bid_price - i * spread * 0.1,
                    size=np.random.exponential(1000),
                    num_orders=np.random.poisson(5)
                ))
                
                ask_levels.append(OrderBookLevel(
                    price=ask_price + i * spread * 0.1,
                    size=np.random.exponential(1000),
                    num_orders=np.random.poisson(5)
                ))
            
            self.order_books[symbol] = OrderBook(
                timestamp=self.current_time,
                symbol=symbol,
                bids=bid_levels,
                asks=ask_levels
            )
            
            # Create market data
            self.market_data[symbol] = MarketData(
                timestamp=self.current_time,
                symbol=symbol,
                bid=bid_price,
                ask=ask_price,
                bid_size=bid_levels[0].size,
                ask_size=ask_levels[0].size,
                last_price=price,
                volume=0.0,
                volatility=self.volatilities[symbol]
            )
    
    def update_market_data(self, time_step: float = 1.0):
        """Update market data with realistic price movements"""
        
        self.current_time += timedelta(seconds=time_step)
        
        for symbol in self.symbols:
            # Generate price movement
            vol = self.volatilities[symbol]
            drift = 0.0  # Assume no drift for simplicity
            
            # Add mean reversion
            current_price = self.prices[symbol]
            long_term_price = np.mean(list(self.price_history[symbol])) if self.price_history[symbol] else current_price
            mean_reversion = -0.01 * (current_price - long_term_price)
            
            # Price change
            price_change = (drift + mean_reversion) * time_step + vol * np.sqrt(time_step) * np.random.normal()
            new_price = current_price * (1 + price_change)
            
            # Update price
            self.prices[symbol] = new_price
            self.price_history[symbol].append(new_price)
            
            # Update spread based on volatility
            base_spread = self.spreads[symbol]
            vol_adjustment = 1 + 2 * abs(price_change) / vol  # Spread widens with volatility
            current_spread = base_spread * vol_adjustment
            
            # Update order book
            self._update_order_book(symbol, new_price, current_spread)
            
            # Update market data
            order_book = self.order_books[symbol]
            self.market_data[symbol] = MarketData(
                timestamp=self.current_time,
                symbol=symbol,
                bid=order_book.best_bid(),
                ask=order_book.best_ask(),
                bid_size=order_book.bids[0].size,
                ask_size=order_book.asks[0].size,
                last_price=new_price,
                volume=self._calculate_volume(symbol),
                volatility=vol
            )
    
    def _update_order_book(self, symbol: str, price: float, spread: float):
        """Update order book with new price and spread"""
        
        bid_price = price - spread / 2
        ask_price = price + spread / 2
        
        # Update order book levels
        order_book = self.order_books[symbol]
        
        # Update bids
        for i, level in enumerate(order_book.bids):
            level.price = bid_price - i * spread * 0.1
            # Simulate liquidity changes
            level.size *= np.random.uniform(0.9, 1.1)
        
        # Update asks
        for i, level in enumerate(order_book.asks):
            level.price = ask_price + i * spread * 0.1
            # Simulate liquidity changes
            level.size *= np.random.uniform(0.9, 1.1)
        
        order_book.timestamp = self.current_time
    
    def _calculate_volume(self, symbol: str) -> float:
        """Calculate trading volume based on price movement"""
        
        if len(self.price_history[symbol]) < 2:
            return 0.0
        
        # Volume increases with price movement
        price_change = abs(self.price_history[symbol][-1] - self.price_history[symbol][-2])
        relative_change = price_change / self.price_history[symbol][-2]
        
        base_volume = 10000  # Base volume
        volume_multiplier = 1 + 10 * relative_change
        
        volume = base_volume * volume_multiplier * np.random.exponential(1)
        self.volume_history[symbol].append(volume)
        
        return volume
    
    def submit_order(self, order: Order) -> str:
        """Submit order to market simulator"""
        
        self.pending_orders[order.order_id] = order
        self.execution_stats['total_orders'] += 1
        
        # Simulate order processing latency
        latency_ms = self._calculate_latency(order)
        
        # Process order after latency
        asyncio.create_task(self._process_order_with_delay(order, latency_ms))
        
        return order.order_id
    
    async def _process_order_with_delay(self, order: Order, latency_ms: float):
        """Process order after simulated latency"""
        
        await asyncio.sleep(latency_ms / 1000.0)  # Convert to seconds
        self._process_order(order)
    
    def _process_order(self, order: Order):
        """Process order execution"""
        
        if order.order_id not in self.pending_orders:
            return
        
        symbol = order.symbol
        order_book = self.order_books[symbol]
        
        if order.order_type == OrderType.MARKET:
            self._execute_market_order(order, order_book)
        elif order.order_type == OrderType.LIMIT:
            self._execute_limit_order(order, order_book)
        else:
            logger.warning(f"Order type {order.order_type} not implemented")
    
    def _execute_market_order(self, order: Order, order_book: OrderBook):
        """Execute market order with realistic fills"""
        
        remaining_qty = order.quantity - order.filled_quantity
        
        if remaining_qty <= 0:
            return
        
        # Determine execution side
        if order.side == OrderSide.BUY:
            levels = order_book.asks
        else:
            levels = order_book.bids
        
        # Execute against order book levels
        total_filled = 0.0
        total_cost = 0.0
        
        for level in levels:
            if remaining_qty <= 0:
                break
            
            # Calculate available liquidity
            available_qty = min(level.size, remaining_qty)
            
            # Apply market impact
            impact_price = self._apply_market_impact(
                order.symbol, level.price, available_qty, order.side
            )
            
            # Create fill
            fill = Fill(
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=available_qty,
                price=impact_price,
                timestamp=self.current_time,
                commission=self._calculate_commission(available_qty, impact_price),
                market_impact=abs(impact_price - level.price),
                latency_ms=self._calculate_latency(order)
            )
            
            # Update order
            order.fills.append(fill.__dict__)
            order.filled_quantity += available_qty
            total_filled += available_qty
            total_cost += available_qty * impact_price
            
            # Update order book
            level.size -= available_qty
            
            # Store fill
            self.fills.append(fill)
            
            remaining_qty -= available_qty
        
        # Update order status
        if order.filled_quantity >= order.quantity:
            order.status = "FILLED"
            order.avg_fill_price = total_cost / order.filled_quantity
            self.filled_orders[order.order_id] = order
            del self.pending_orders[order.order_id]
        else:
            order.status = "PARTIAL_FILL"
        
        # Update execution stats
        self.execution_stats['total_fills'] += 1
        self._update_execution_stats(order)
    
    def _execute_limit_order(self, order: Order, order_book: OrderBook):
        """Execute limit order if price is favorable"""
        
        if order.side == OrderSide.BUY:
            best_ask = order_book.best_ask()
            if best_ask and order.price >= best_ask:
                # Convert to market order for execution
                market_order = Order(
                    order_id=order.order_id,
                    symbol=order.symbol,
                    side=order.side,
                    order_type=OrderType.MARKET,
                    quantity=order.quantity,
                    timestamp=order.timestamp
                )
                self._execute_market_order(market_order, order_book)
        else:
            best_bid = order_book.best_bid()
            if best_bid and order.price <= best_bid:
                # Convert to market order for execution
                market_order = Order(
                    order_id=order.order_id,
                    symbol=order.symbol,
                    side=order.side,
                    order_type=OrderType.MARKET,
                    quantity=order.quantity,
                    timestamp=order.timestamp
                )
                self._execute_market_order(market_order, order_book)
    
    def _apply_market_impact(
        self,
        symbol: str,
        base_price: float,
        quantity: float,
        side: OrderSide
    ) -> float:
        """Apply market impact to execution price"""
        
        # Calculate average daily volume
        avg_volume = np.mean(list(self.volume_history[symbol])) if self.volume_history[symbol] else 100000
        
        # Calculate participation rate
        participation_rate = quantity / avg_volume
        
        # Apply impact model
        if self.market_impact_model.model_type == "linear":
            impact = self.market_impact_model.temporary_impact * participation_rate
        elif self.market_impact_model.model_type == "square_root":
            impact = self.market_impact_model.temporary_impact * np.sqrt(participation_rate)
        elif self.market_impact_model.model_type == "power_law":
            impact = self.market_impact_model.temporary_impact * (participation_rate ** self.market_impact_model.alpha)
        else:
            impact = 0.0
        
        # Convert to price impact
        price_impact = base_price * impact / 10000  # Convert basis points to price
        
        # Apply directional impact
        if side == OrderSide.BUY:
            return base_price + price_impact
        else:
            return base_price - price_impact
    
    def _calculate_latency(self, order: Order) -> float:
        """Calculate order processing latency"""
        
        base_latency = self.latency_model['base_latency_ms']
        network_jitter = np.random.normal(0, self.latency_model['network_jitter_ms'])
        processing_delay = self.latency_model['processing_delay_ms']
        
        # Queue delay based on market volatility
        volatility = self.volatilities[order.symbol]
        queue_delay = self.latency_model['queue_delay_factor'] * volatility * 1000
        
        total_latency = base_latency + network_jitter + processing_delay + queue_delay
        return max(0.1, total_latency)  # Minimum 0.1ms
    
    def _calculate_commission(self, quantity: float, price: float) -> float:
        """Calculate trading commission"""
        
        # Simple commission model
        notional = quantity * price
        commission_rate = 0.0005  # 5 basis points
        
        return notional * commission_rate
    
    def _update_execution_stats(self, order: Order):
        """Update execution statistics"""
        
        if order.fills:
            # Calculate average fill time
            fill_times = [fill.get('latency_ms', 0) for fill in order.fills]
            avg_fill_time = np.mean(fill_times)
            
            # Calculate slippage
            if order.side == OrderSide.BUY:
                reference_price = self.market_data[order.symbol].ask
            else:
                reference_price = self.market_data[order.symbol].bid
            
            slippage = abs(order.avg_fill_price - reference_price) / reference_price * 10000  # Basis points
            
            # Calculate market impact
            impact_costs = [fill.get('market_impact', 0) for fill in order.fills]
            avg_impact = np.mean(impact_costs) / reference_price * 10000  # Basis points
            
            # Update running averages
            total_fills = self.execution_stats['total_fills']
            self.execution_stats['avg_fill_time_ms'] = (
                (self.execution_stats['avg_fill_time_ms'] * (total_fills - 1) + avg_fill_time) / total_fills
            )
            self.execution_stats['avg_slippage_bps'] = (
                (self.execution_stats['avg_slippage_bps'] * (total_fills - 1) + slippage) / total_fills
            )
            self.execution_stats['avg_market_impact_bps'] = (
                (self.execution_stats['avg_market_impact_bps'] * (total_fills - 1) + avg_impact) / total_fills
            )
    
    def get_market_data(self, symbol: str) -> MarketData:
        """Get current market data for symbol"""
        return self.market_data.get(symbol)
    
    def get_order_book(self, symbol: str) -> OrderBook:
        """Get current order book for symbol"""
        return self.order_books.get(symbol)
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        return self.execution_stats.copy()
    
    def get_fills(self, order_id: Optional[str] = None) -> List[Fill]:
        """Get fills for specific order or all fills"""
        if order_id:
            return [fill for fill in self.fills if fill.order_id == order_id]
        return self.fills.copy()
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order"""
        if order_id in self.pending_orders:
            order = self.pending_orders[order_id]
            order.status = "CANCELED"
            del self.pending_orders[order_id]
            return True
        return False
    
    def get_order_status(self, order_id: str) -> Optional[str]:
        """Get order status"""
        if order_id in self.pending_orders:
            return self.pending_orders[order_id].status
        elif order_id in self.filled_orders:
            return self.filled_orders[order_id].status
        return None


@jit(nopython=True)
def calculate_vwap_price(prices: np.ndarray, volumes: np.ndarray) -> float:
    """Calculate Volume Weighted Average Price (VWAP)"""
    if len(prices) == 0 or len(volumes) == 0:
        return 0.0
    
    total_volume = np.sum(volumes)
    if total_volume == 0:
        return np.mean(prices)
    
    return np.sum(prices * volumes) / total_volume


@jit(nopython=True)
def calculate_twap_price(prices: np.ndarray) -> float:
    """Calculate Time Weighted Average Price (TWAP)"""
    if len(prices) == 0:
        return 0.0
    
    return np.mean(prices)


def simulate_realistic_trading_day(
    symbols: List[str],
    initial_prices: Dict[str, float],
    volatilities: Dict[str, float],
    duration_hours: float = 8.0,
    time_step_seconds: float = 1.0
) -> MarketSimulator:
    """
    Simulate a realistic trading day with market data
    
    Args:
        symbols: List of symbols to simulate
        initial_prices: Initial prices for each symbol
        volatilities: Volatilities for each symbol
        duration_hours: Duration of simulation in hours
        time_step_seconds: Time step for simulation
        
    Returns:
        MarketSimulator instance with historical data
    """
    
    # Calculate spreads based on volatility
    spreads = {symbol: max(0.01, vol * 0.1) for symbol, vol in volatilities.items()}
    
    # Create market simulator
    simulator = MarketSimulator(
        symbols=symbols,
        initial_prices=initial_prices,
        volatilities=volatilities,
        spreads=spreads,
        market_impact_model=MarketImpactModel(
            model_type="square_root",
            temporary_impact=0.002,
            permanent_impact=0.001
        )
    )
    
    # Run simulation
    total_steps = int(duration_hours * 3600 / time_step_seconds)
    
    for step in range(total_steps):
        simulator.update_market_data(time_step_seconds)
    
    return simulator


class TransactionCostAnalyzer:
    """Analyze transaction costs from trading simulation"""
    
    def __init__(self, fills: List[Fill]):
        self.fills = fills
    
    def analyze_costs(self) -> Dict[str, float]:
        """Analyze transaction costs"""
        
        if not self.fills:
            return {}
        
        total_notional = sum(fill.quantity * fill.price for fill in self.fills)
        total_commission = sum(fill.commission for fill in self.fills)
        total_market_impact = sum(fill.market_impact * fill.quantity for fill in self.fills)
        
        # Calculate cost ratios
        commission_bps = (total_commission / total_notional) * 10000
        market_impact_bps = (total_market_impact / total_notional) * 10000
        total_cost_bps = commission_bps + market_impact_bps
        
        # Calculate latency stats
        latencies = [fill.latency_ms for fill in self.fills]
        
        return {
            'total_notional': total_notional,
            'total_commission': total_commission,
            'total_market_impact': total_market_impact,
            'commission_bps': commission_bps,
            'market_impact_bps': market_impact_bps,
            'total_cost_bps': total_cost_bps,
            'avg_latency_ms': np.mean(latencies),
            'max_latency_ms': np.max(latencies),
            'latency_std_ms': np.std(latencies),
            'fill_count': len(self.fills)
        }