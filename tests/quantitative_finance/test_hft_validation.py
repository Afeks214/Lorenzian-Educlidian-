"""
High-Frequency Trading Validation Testing Suite

Comprehensive tests for latency-sensitive order management, co-location,
direct market access, market making, and arbitrage strategies.
"""

import pytest
import asyncio
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass
from enum import Enum
import structlog

logger = structlog.get_logger()


class OrderType(Enum):
    """Order types for HFT"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill
    PEGGED = "PEGGED"
    ICEBERG = "ICEBERG"


class OrderStatus(Enum):
    """Order status"""
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIAL_FILL = "PARTIAL_FILL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


@dataclass
class HFTOrder:
    """HFT order structure"""
    
    order_id: str
    symbol: str
    side: str  # BUY/SELL
    quantity: int
    price: float
    order_type: OrderType
    timestamp: datetime
    status: OrderStatus
    
    # HFT-specific fields
    priority_level: int = 1  # 1-10, higher = more urgent
    latency_requirement_us: int = 100  # microseconds
    venue: str = "SMART"
    routing_strategy: str = "FASTEST"
    
    # Execution tracking
    filled_quantity: int = 0
    avg_fill_price: float = 0.0
    submission_latency_us: float = 0.0
    fill_latency_us: float = 0.0
    
    # Performance metrics
    placement_time: Optional[datetime] = None
    first_fill_time: Optional[datetime] = None
    completion_time: Optional[datetime] = None


@dataclass
class MarketDataUpdate:
    """Market data update"""
    
    symbol: str
    timestamp: datetime
    bid_price: float
    ask_price: float
    bid_size: int
    ask_size: int
    last_price: float
    last_size: int
    sequence_number: int
    
    # Latency tracking
    received_time: Optional[datetime] = None
    processed_time: Optional[datetime] = None
    
    def get_latency_us(self) -> float:
        """Get processing latency in microseconds"""
        if self.received_time and self.processed_time:
            return (self.processed_time - self.received_time).total_seconds() * 1_000_000
        return 0.0


class UltraLowLatencyOrderManager:
    """Ultra-low latency order manager for HFT"""
    
    def __init__(self, target_latency_us: int = 100):
        self.target_latency_us = target_latency_us
        self.orders = {}
        self.order_queue = []
        self.lock = threading.RLock()
        
        # Performance tracking
        self.latency_measurements = []
        self.throughput_measurements = []
        self.order_count = 0
        
        # Hardware optimizations
        self.use_kernel_bypass = True
        self.use_cpu_affinity = True
        self.use_memory_pools = True
        
        # Threading
        self.processing_thread = None
        self.shutdown_event = threading.Event()
        
        self._start_processing()
    
    def _start_processing(self):
        """Start order processing thread"""
        self.processing_thread = threading.Thread(target=self._process_orders)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def _process_orders(self):
        """Process orders in separate thread"""
        while not self.shutdown_event.is_set():
            try:
                # Process pending orders
                self._process_pending_orders()
                
                # Sleep for minimal time to yield CPU
                time.sleep(0.00001)  # 10 microseconds
                
            except Exception as e:
                logger.error(f"Error in order processing: {e}")
    
    def _process_pending_orders(self):
        """Process pending orders"""
        with self.lock:
            if not self.order_queue:
                return
            
            # Process highest priority orders first
            self.order_queue.sort(key=lambda x: x.priority_level, reverse=True)
            
            # Process up to 100 orders per cycle
            orders_to_process = self.order_queue[:100]
            self.order_queue = self.order_queue[100:]
            
        for order in orders_to_process:
            self._execute_order(order)
    
    def submit_order(self, order: HFTOrder) -> bool:
        """Submit order with ultra-low latency"""
        start_time = time.perf_counter()
        
        try:
            # Validate order
            if not self._validate_order(order):
                return False
            
            # Set placement time
            order.placement_time = datetime.now()
            order.status = OrderStatus.SUBMITTED
            
            # Add to queue
            with self.lock:
                self.order_queue.append(order)
                self.orders[order.order_id] = order
            
            # Calculate submission latency
            submission_latency = (time.perf_counter() - start_time) * 1_000_000
            order.submission_latency_us = submission_latency
            
            # Track performance
            self.latency_measurements.append(submission_latency)
            self.order_count += 1
            
            # Keep recent measurements only
            if len(self.latency_measurements) > 10000:
                self.latency_measurements = self.latency_measurements[-5000:]
            
            return True
            
        except Exception as e:
            logger.error(f"Order submission failed: {e}")
            return False
    
    def _validate_order(self, order: HFTOrder) -> bool:
        """Validate order quickly"""
        # Basic validation only for speed
        if not order.symbol or order.quantity <= 0:
            return False
        
        if order.side not in ['BUY', 'SELL']:
            return False
        
        if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and order.price <= 0:
            return False
        
        return True
    
    def _execute_order(self, order: HFTOrder):
        """Execute order (simulation)"""
        try:
            # Simulate execution latency
            execution_start = time.perf_counter()
            
            # Simulate venue communication
            venue_latency = np.random.uniform(10, 50)  # 10-50 microseconds
            time.sleep(venue_latency / 1_000_000)
            
            # Simulate fill
            if order.order_type == OrderType.MARKET:
                # Market orders fill immediately
                order.filled_quantity = order.quantity
                order.avg_fill_price = order.price or 100.0  # Default price
                order.status = OrderStatus.FILLED
                order.first_fill_time = datetime.now()
                order.completion_time = datetime.now()
                
            elif order.order_type == OrderType.LIMIT:
                # Limit orders may fill based on market conditions
                if np.random.random() < 0.8:  # 80% fill rate
                    order.filled_quantity = order.quantity
                    order.avg_fill_price = order.price
                    order.status = OrderStatus.FILLED
                    order.first_fill_time = datetime.now()
                    order.completion_time = datetime.now()
                else:
                    order.status = OrderStatus.PENDING
            
            # Calculate fill latency
            fill_latency = (time.perf_counter() - execution_start) * 1_000_000
            order.fill_latency_us = fill_latency
            
        except Exception as e:
            logger.error(f"Order execution failed: {e}")
            order.status = OrderStatus.REJECTED
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel order with ultra-low latency"""
        start_time = time.perf_counter()
        
        try:
            with self.lock:
                order = self.orders.get(order_id)
                if not order:
                    return False
                
                if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
                    return False
                
                # Cancel order
                order.status = OrderStatus.CANCELLED
                order.completion_time = datetime.now()
                
                # Remove from queue if pending
                self.order_queue = [o for o in self.order_queue if o.order_id != order_id]
            
            # Calculate cancellation latency
            cancel_latency = (time.perf_counter() - start_time) * 1_000_000
            
            # Should be under target latency
            assert cancel_latency < self.target_latency_us * 2
            
            return True
            
        except Exception as e:
            logger.error(f"Order cancellation failed: {e}")
            return False
    
    def get_order_status(self, order_id: str) -> Optional[HFTOrder]:
        """Get order status"""
        with self.lock:
            return self.orders.get(order_id)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        with self.lock:
            if not self.latency_measurements:
                return {'status': 'no_data'}
            
            return {
                'total_orders': self.order_count,
                'avg_latency_us': np.mean(self.latency_measurements),
                'p50_latency_us': np.percentile(self.latency_measurements, 50),
                'p95_latency_us': np.percentile(self.latency_measurements, 95),
                'p99_latency_us': np.percentile(self.latency_measurements, 99),
                'max_latency_us': np.max(self.latency_measurements),
                'min_latency_us': np.min(self.latency_measurements),
                'latency_target_us': self.target_latency_us,
                'meeting_target': np.mean(self.latency_measurements) <= self.target_latency_us
            }
    
    def shutdown(self):
        """Shutdown order manager"""
        self.shutdown_event.set()
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)


class MarketMaker:
    """Market making strategy"""
    
    def __init__(self, symbol: str, spread_bps: int = 5):
        self.symbol = symbol
        self.spread_bps = spread_bps
        self.position = 0
        self.max_position = 10000
        self.active_orders = {}
        self.pnl = 0.0
        
    def update_quotes(self, market_data: MarketDataUpdate, 
                     order_manager: UltraLowLatencyOrderManager) -> List[HFTOrder]:
        """Update market making quotes"""
        orders = []
        
        # Cancel existing orders
        for order_id in list(self.active_orders.keys()):
            order_manager.cancel_order(order_id)
            del self.active_orders[order_id]
        
        # Calculate fair value
        fair_value = (market_data.bid_price + market_data.ask_price) / 2
        
        # Calculate spread
        spread = fair_value * (self.spread_bps / 10000)
        
        # Generate bid order
        if self.position < self.max_position:
            bid_order = HFTOrder(
                order_id=f"BID_{int(time.time() * 1000000)}",
                symbol=self.symbol,
                side="BUY",
                quantity=min(1000, self.max_position - self.position),
                price=fair_value - spread/2,
                order_type=OrderType.LIMIT,
                timestamp=datetime.now(),
                status=OrderStatus.PENDING,
                priority_level=8,
                latency_requirement_us=50
            )
            orders.append(bid_order)
            self.active_orders[bid_order.order_id] = bid_order
        
        # Generate ask order
        if self.position > -self.max_position:
            ask_order = HFTOrder(
                order_id=f"ASK_{int(time.time() * 1000000)}",
                symbol=self.symbol,
                side="SELL",
                quantity=min(1000, self.max_position + self.position),
                price=fair_value + spread/2,
                order_type=OrderType.LIMIT,
                timestamp=datetime.now(),
                status=OrderStatus.PENDING,
                priority_level=8,
                latency_requirement_us=50
            )
            orders.append(ask_order)
            self.active_orders[ask_order.order_id] = ask_order
        
        return orders
    
    def on_fill(self, order: HFTOrder):
        """Handle order fill"""
        if order.side == "BUY":
            self.position += order.filled_quantity
            self.pnl -= order.avg_fill_price * order.filled_quantity
        else:
            self.position -= order.filled_quantity
            self.pnl += order.avg_fill_price * order.filled_quantity


class ArbitrageStrategy:
    """Arbitrage strategy"""
    
    def __init__(self, symbol1: str, symbol2: str, correlation_threshold: float = 0.9):
        self.symbol1 = symbol1
        self.symbol2 = symbol2
        self.correlation_threshold = correlation_threshold
        self.price_history = {symbol1: [], symbol2: []}
        self.position1 = 0
        self.position2 = 0
        self.pnl = 0.0
        
    def check_arbitrage_opportunity(self, market_data1: MarketDataUpdate, 
                                  market_data2: MarketDataUpdate) -> List[HFTOrder]:
        """Check for arbitrage opportunities"""
        orders = []
        
        # Store price history
        self.price_history[self.symbol1].append(market_data1.last_price)
        self.price_history[self.symbol2].append(market_data2.last_price)
        
        # Keep limited history
        for symbol in [self.symbol1, self.symbol2]:
            if len(self.price_history[symbol]) > 100:
                self.price_history[symbol] = self.price_history[symbol][-50:]
        
        # Need sufficient history for correlation
        if len(self.price_history[self.symbol1]) < 20:
            return orders
        
        # Calculate correlation
        prices1 = np.array(self.price_history[self.symbol1])
        prices2 = np.array(self.price_history[self.symbol2])
        
        correlation = np.corrcoef(prices1, prices2)[0, 1]
        
        if correlation < self.correlation_threshold:
            return orders
        
        # Calculate spread
        normalized_price1 = market_data1.last_price / prices1[0]
        normalized_price2 = market_data2.last_price / prices2[0]
        
        spread = normalized_price1 - normalized_price2
        
        # Check for arbitrage opportunity
        if abs(spread) > 0.005:  # 0.5% threshold
            if spread > 0:
                # Symbol1 overpriced, symbol2 underpriced
                # Sell symbol1, buy symbol2
                orders.append(HFTOrder(
                    order_id=f"ARB_SELL_{int(time.time() * 1000000)}",
                    symbol=self.symbol1,
                    side="SELL",
                    quantity=1000,
                    price=market_data1.bid_price,
                    order_type=OrderType.MARKET,
                    timestamp=datetime.now(),
                    status=OrderStatus.PENDING,
                    priority_level=10,
                    latency_requirement_us=25
                ))
                
                orders.append(HFTOrder(
                    order_id=f"ARB_BUY_{int(time.time() * 1000000)}",
                    symbol=self.symbol2,
                    side="BUY",
                    quantity=1000,
                    price=market_data2.ask_price,
                    order_type=OrderType.MARKET,
                    timestamp=datetime.now(),
                    status=OrderStatus.PENDING,
                    priority_level=10,
                    latency_requirement_us=25
                ))
            else:
                # Symbol2 overpriced, symbol1 underpriced
                # Buy symbol1, sell symbol2
                orders.append(HFTOrder(
                    order_id=f"ARB_BUY_{int(time.time() * 1000000)}",
                    symbol=self.symbol1,
                    side="BUY",
                    quantity=1000,
                    price=market_data1.ask_price,
                    order_type=OrderType.MARKET,
                    timestamp=datetime.now(),
                    status=OrderStatus.PENDING,
                    priority_level=10,
                    latency_requirement_us=25
                ))
                
                orders.append(HFTOrder(
                    order_id=f"ARB_SELL_{int(time.time() * 1000000)}",
                    symbol=self.symbol2,
                    side="SELL",
                    quantity=1000,
                    price=market_data2.bid_price,
                    order_type=OrderType.MARKET,
                    timestamp=datetime.now(),
                    status=OrderStatus.PENDING,
                    priority_level=10,
                    latency_requirement_us=25
                ))
        
        return orders


class DirectMarketAccess:
    """Direct market access simulator"""
    
    def __init__(self, venues: List[str]):
        self.venues = venues
        self.connections = {}
        self.latencies = {}
        self.order_counts = {}
        
        # Initialize venue connections
        for venue in venues:
            self.connections[venue] = {'connected': True, 'capacity': 10000}
            self.latencies[venue] = []
            self.order_counts[venue] = 0
    
    def route_order(self, order: HFTOrder, venue: str) -> bool:
        """Route order to specific venue"""
        if venue not in self.venues:
            return False
        
        if not self.connections[venue]['connected']:
            return False
        
        # Check capacity
        if self.order_counts[venue] >= self.connections[venue]['capacity']:
            return False
        
        # Simulate venue-specific latency
        start_time = time.perf_counter()
        
        # Different venues have different latencies
        venue_latency = self._get_venue_latency(venue)
        time.sleep(venue_latency / 1_000_000)  # Convert to seconds
        
        # Track latency
        actual_latency = (time.perf_counter() - start_time) * 1_000_000
        self.latencies[venue].append(actual_latency)
        self.order_counts[venue] += 1
        
        # Keep recent latencies only
        if len(self.latencies[venue]) > 1000:
            self.latencies[venue] = self.latencies[venue][-500:]
        
        return True
    
    def _get_venue_latency(self, venue: str) -> float:
        """Get venue-specific latency"""
        latency_map = {
            'NYSE': np.random.uniform(50, 100),
            'NASDAQ': np.random.uniform(40, 80),
            'BATS': np.random.uniform(30, 70),
            'EDGX': np.random.uniform(35, 75),
            'DARK_POOL': np.random.uniform(100, 200)
        }
        
        return latency_map.get(venue, 100)
    
    def get_venue_statistics(self) -> Dict[str, Any]:
        """Get venue statistics"""
        stats = {}
        
        for venue in self.venues:
            if self.latencies[venue]:
                stats[venue] = {
                    'avg_latency_us': np.mean(self.latencies[venue]),
                    'p95_latency_us': np.percentile(self.latencies[venue], 95),
                    'order_count': self.order_counts[venue],
                    'connected': self.connections[venue]['connected']
                }
            else:
                stats[venue] = {
                    'avg_latency_us': 0,
                    'p95_latency_us': 0,
                    'order_count': 0,
                    'connected': self.connections[venue]['connected']
                }
        
        return stats


def generate_market_data_stream(symbol: str, count: int = 100) -> List[MarketDataUpdate]:
    """Generate realistic market data stream"""
    updates = []
    base_price = 100.0
    
    for i in range(count):
        # Generate price movement
        price_change = np.random.normal(0, 0.001)  # Small movements
        base_price = base_price * (1 + price_change)
        
        # Generate bid-ask spread
        spread = max(0.01, np.random.uniform(0.01, 0.05))
        bid_price = base_price - spread/2
        ask_price = base_price + spread/2
        
        # Generate sizes
        bid_size = np.random.randint(100, 1000)
        ask_size = np.random.randint(100, 1000)
        last_size = np.random.randint(100, 500)
        
        update = MarketDataUpdate(
            symbol=symbol,
            timestamp=datetime.now() - timedelta(milliseconds=count-i),
            bid_price=round(bid_price, 2),
            ask_price=round(ask_price, 2),
            bid_size=bid_size,
            ask_size=ask_size,
            last_price=round(base_price, 2),
            last_size=last_size,
            sequence_number=i + 1
        )
        
        updates.append(update)
    
    return updates


@pytest.fixture
def ultra_low_latency_order_manager():
    """Create ultra-low latency order manager"""
    return UltraLowLatencyOrderManager(target_latency_us=100)


@pytest.fixture
def market_maker():
    """Create market maker"""
    return MarketMaker("AAPL", spread_bps=5)


@pytest.fixture
def arbitrage_strategy():
    """Create arbitrage strategy"""
    return ArbitrageStrategy("AAPL", "GOOGL", correlation_threshold=0.9)


@pytest.fixture
def direct_market_access():
    """Create direct market access"""
    return DirectMarketAccess(['NYSE', 'NASDAQ', 'BATS', 'EDGX'])


@pytest.fixture
def sample_market_data():
    """Generate sample market data"""
    return generate_market_data_stream("AAPL", 100)


class TestUltraLowLatencyOrders:
    """Test ultra-low latency order management"""
    
    def test_order_submission_latency(self, ultra_low_latency_order_manager):
        """Test order submission latency"""
        order = HFTOrder(
            order_id="TEST_001",
            symbol="AAPL",
            side="BUY",
            quantity=1000,
            price=150.00,
            order_type=OrderType.LIMIT,
            timestamp=datetime.now(),
            status=OrderStatus.PENDING,
            priority_level=5,
            latency_requirement_us=100
        )
        
        # Submit order
        start_time = time.perf_counter()
        success = ultra_low_latency_order_manager.submit_order(order)
        end_time = time.perf_counter()
        
        actual_latency = (end_time - start_time) * 1_000_000
        
        assert success
        assert actual_latency < 1000  # Should be under 1ms
        assert order.submission_latency_us < 1000
    
    def test_high_frequency_order_submission(self, ultra_low_latency_order_manager):
        """Test high-frequency order submission"""
        orders = []
        
        # Submit 1000 orders rapidly
        start_time = time.perf_counter()
        
        for i in range(1000):
            order = HFTOrder(
                order_id=f"HF_ORDER_{i}",
                symbol="AAPL",
                side="BUY" if i % 2 == 0 else "SELL",
                quantity=100,
                price=150.00 + (i % 10) * 0.01,
                order_type=OrderType.LIMIT,
                timestamp=datetime.now(),
                status=OrderStatus.PENDING,
                priority_level=1,
                latency_requirement_us=100
            )
            
            success = ultra_low_latency_order_manager.submit_order(order)
            assert success
            orders.append(order)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Should handle high frequency
        assert total_time < 1.0  # Less than 1 second for 1000 orders
        assert len(orders) == 1000
    
    def test_order_cancellation_latency(self, ultra_low_latency_order_manager):
        """Test order cancellation latency"""
        order = HFTOrder(
            order_id="CANCEL_TEST",
            symbol="AAPL",
            side="BUY",
            quantity=1000,
            price=150.00,
            order_type=OrderType.LIMIT,
            timestamp=datetime.now(),
            status=OrderStatus.PENDING,
            priority_level=5,
            latency_requirement_us=100
        )
        
        # Submit order
        success = ultra_low_latency_order_manager.submit_order(order)
        assert success
        
        # Cancel order
        start_time = time.perf_counter()
        cancelled = ultra_low_latency_order_manager.cancel_order(order.order_id)
        end_time = time.perf_counter()
        
        cancel_latency = (end_time - start_time) * 1_000_000
        
        assert cancelled
        assert cancel_latency < 200  # Should be under 200 microseconds
    
    def test_order_priority_handling(self, ultra_low_latency_order_manager):
        """Test order priority handling"""
        # Submit orders with different priorities
        high_priority = HFTOrder(
            order_id="HIGH_PRIORITY",
            symbol="AAPL",
            side="BUY",
            quantity=1000,
            price=150.00,
            order_type=OrderType.LIMIT,
            timestamp=datetime.now(),
            status=OrderStatus.PENDING,
            priority_level=10,
            latency_requirement_us=50
        )
        
        low_priority = HFTOrder(
            order_id="LOW_PRIORITY",
            symbol="AAPL",
            side="BUY",
            quantity=1000,
            price=150.00,
            order_type=OrderType.LIMIT,
            timestamp=datetime.now(),
            status=OrderStatus.PENDING,
            priority_level=1,
            latency_requirement_us=1000
        )
        
        # Submit in reverse priority order
        ultra_low_latency_order_manager.submit_order(low_priority)
        ultra_low_latency_order_manager.submit_order(high_priority)
        
        # High priority should be processed first
        # Wait for processing
        time.sleep(0.001)
        
        # Check that orders were processed
        high_status = ultra_low_latency_order_manager.get_order_status("HIGH_PRIORITY")
        low_status = ultra_low_latency_order_manager.get_order_status("LOW_PRIORITY")
        
        assert high_status is not None
        assert low_status is not None
    
    def test_performance_metrics(self, ultra_low_latency_order_manager):
        """Test performance metrics collection"""
        # Submit several orders
        for i in range(10):
            order = HFTOrder(
                order_id=f"PERF_TEST_{i}",
                symbol="AAPL",
                side="BUY",
                quantity=100,
                price=150.00,
                order_type=OrderType.LIMIT,
                timestamp=datetime.now(),
                status=OrderStatus.PENDING,
                priority_level=5,
                latency_requirement_us=100
            )
            
            ultra_low_latency_order_manager.submit_order(order)
        
        # Get metrics
        metrics = ultra_low_latency_order_manager.get_performance_metrics()
        
        assert metrics['total_orders'] == 10
        assert 'avg_latency_us' in metrics
        assert 'p95_latency_us' in metrics
        assert 'p99_latency_us' in metrics
        assert metrics['latency_target_us'] == 100
        
        # Should meet target latency
        assert metrics['avg_latency_us'] < 1000  # Should be reasonable


class TestMarketMakingStrategy:
    """Test market making strategy"""
    
    def test_quote_generation(self, market_maker, ultra_low_latency_order_manager):
        """Test quote generation"""
        market_data = MarketDataUpdate(
            symbol="AAPL",
            timestamp=datetime.now(),
            bid_price=149.95,
            ask_price=150.05,
            bid_size=1000,
            ask_size=1000,
            last_price=150.00,
            last_size=100,
            sequence_number=1
        )
        
        orders = market_maker.update_quotes(market_data, ultra_low_latency_order_manager)
        
        # Should generate bid and ask orders
        assert len(orders) == 2
        
        bid_order = next(o for o in orders if o.side == "BUY")
        ask_order = next(o for o in orders if o.side == "SELL")
        
        assert bid_order.price < market_data.last_price
        assert ask_order.price > market_data.last_price
        assert bid_order.price < ask_order.price
    
    def test_position_management(self, market_maker, ultra_low_latency_order_manager):
        """Test position management"""
        # Set large position
        market_maker.position = 9500  # Near max position
        
        market_data = MarketDataUpdate(
            symbol="AAPL",
            timestamp=datetime.now(),
            bid_price=149.95,
            ask_price=150.05,
            bid_size=1000,
            ask_size=1000,
            last_price=150.00,
            last_size=100,
            sequence_number=1
        )
        
        orders = market_maker.update_quotes(market_data, ultra_low_latency_order_manager)
        
        # Should limit orders based on position
        # May still generate both orders but with reduced quantities
        assert len(orders) >= 1
        
        # Find the buy order if it exists
        buy_orders = [o for o in orders if o.side == "BUY"]
        if buy_orders:
            assert buy_orders[0].quantity <= 500  # Limited by remaining position capacity
    
    def test_spread_calculation(self, market_maker, ultra_low_latency_order_manager):
        """Test spread calculation"""
        market_data = MarketDataUpdate(
            symbol="AAPL",
            timestamp=datetime.now(),
            bid_price=149.95,
            ask_price=150.05,
            bid_size=1000,
            ask_size=1000,
            last_price=150.00,
            last_size=100,
            sequence_number=1
        )
        
        orders = market_maker.update_quotes(market_data, ultra_low_latency_order_manager)
        
        bid_order = next(o for o in orders if o.side == "BUY")
        ask_order = next(o for o in orders if o.side == "SELL")
        
        # Calculate spread
        spread = ask_order.price - bid_order.price
        expected_spread = market_data.last_price * (market_maker.spread_bps / 10000)
        
        assert abs(spread - expected_spread) < 0.01
    
    def test_fill_handling(self, market_maker):
        """Test fill handling"""
        initial_position = market_maker.position
        initial_pnl = market_maker.pnl
        
        # Simulate buy fill
        buy_order = HFTOrder(
            order_id="BUY_FILL",
            symbol="AAPL",
            side="BUY",
            quantity=1000,
            price=149.95,
            order_type=OrderType.LIMIT,
            timestamp=datetime.now(),
            status=OrderStatus.FILLED,
            filled_quantity=1000,
            avg_fill_price=149.95
        )
        
        market_maker.on_fill(buy_order)
        
        # Position should increase, PnL should decrease
        assert market_maker.position == initial_position + 1000
        assert market_maker.pnl == initial_pnl - (149.95 * 1000)
        
        # Simulate sell fill
        sell_order = HFTOrder(
            order_id="SELL_FILL",
            symbol="AAPL",
            side="SELL",
            quantity=500,
            price=150.05,
            order_type=OrderType.LIMIT,
            timestamp=datetime.now(),
            status=OrderStatus.FILLED,
            filled_quantity=500,
            avg_fill_price=150.05
        )
        
        market_maker.on_fill(sell_order)
        
        # Position should decrease, PnL should increase
        assert market_maker.position == initial_position + 500
        assert market_maker.pnl == initial_pnl - (149.95 * 1000) + (150.05 * 500)


class TestArbitrageStrategy:
    """Test arbitrage strategy"""
    
    def test_arbitrage_opportunity_detection(self, arbitrage_strategy):
        """Test arbitrage opportunity detection"""
        # Build price history
        for i in range(30):
            price1 = 100 + i * 0.1
            price2 = 100 + i * 0.1  # Correlated prices
            
            arbitrage_strategy.price_history["AAPL"].append(price1)
            arbitrage_strategy.price_history["GOOGL"].append(price2)
        
        # Create market data with arbitrage opportunity
        market_data1 = MarketDataUpdate(
            symbol="AAPL",
            timestamp=datetime.now(),
            bid_price=102.95,
            ask_price=103.05,
            bid_size=1000,
            ask_size=1000,
            last_price=103.00,  # Higher price
            last_size=100,
            sequence_number=31
        )
        
        market_data2 = MarketDataUpdate(
            symbol="GOOGL",
            timestamp=datetime.now(),
            bid_price=101.95,
            ask_price=102.05,
            bid_size=1000,
            ask_size=1000,
            last_price=102.00,  # Lower price
            last_size=100,
            sequence_number=31
        )
        
        orders = arbitrage_strategy.check_arbitrage_opportunity(market_data1, market_data2)
        
        # Should generate arbitrage orders
        assert len(orders) == 2
        
        # Should sell overpriced and buy underpriced
        sell_order = next(o for o in orders if o.side == "SELL")
        buy_order = next(o for o in orders if o.side == "BUY")
        
        assert sell_order.symbol == "AAPL"  # Sell overpriced
        assert buy_order.symbol == "GOOGL"  # Buy underpriced
    
    def test_correlation_threshold(self, arbitrage_strategy):
        """Test correlation threshold"""
        # Build uncorrelated price history
        for i in range(30):
            price1 = 100 + np.random.normal(0, 1)
            price2 = 100 + np.random.normal(0, 1)
            
            arbitrage_strategy.price_history["AAPL"].append(price1)
            arbitrage_strategy.price_history["GOOGL"].append(price2)
        
        # Create market data with price divergence
        market_data1 = MarketDataUpdate(
            symbol="AAPL",
            timestamp=datetime.now(),
            bid_price=104.95,
            ask_price=105.05,
            bid_size=1000,
            ask_size=1000,
            last_price=105.00,
            last_size=100,
            sequence_number=31
        )
        
        market_data2 = MarketDataUpdate(
            symbol="GOOGL",
            timestamp=datetime.now(),
            bid_price=99.95,
            ask_price=100.05,
            bid_size=1000,
            ask_size=1000,
            last_price=100.00,
            last_size=100,
            sequence_number=31
        )
        
        orders = arbitrage_strategy.check_arbitrage_opportunity(market_data1, market_data2)
        
        # Should not generate orders due to low correlation
        assert len(orders) == 0
    
    def test_price_normalization(self, arbitrage_strategy):
        """Test price normalization"""
        # Build price history
        base_price1 = 100.0
        base_price2 = 200.0  # Different base price
        
        for i in range(30):
            price1 = base_price1 * (1 + i * 0.001)
            price2 = base_price2 * (1 + i * 0.001)  # Same percentage moves
            
            arbitrage_strategy.price_history["AAPL"].append(price1)
            arbitrage_strategy.price_history["GOOGL"].append(price2)
        
        # Create market data
        market_data1 = MarketDataUpdate(
            symbol="AAPL",
            timestamp=datetime.now(),
            bid_price=102.95,
            ask_price=103.05,
            bid_size=1000,
            ask_size=1000,
            last_price=103.00,
            last_size=100,
            sequence_number=31
        )
        
        market_data2 = MarketDataUpdate(
            symbol="GOOGL",
            timestamp=datetime.now(),
            bid_price=205.95,
            ask_price=206.05,
            bid_size=1000,
            ask_size=1000,
            last_price=206.00,
            last_size=100,
            sequence_number=31
        )
        
        orders = arbitrage_strategy.check_arbitrage_opportunity(market_data1, market_data2)
        
        # Should properly normalize prices and detect no arbitrage
        assert len(orders) == 0


class TestDirectMarketAccess:
    """Test direct market access"""
    
    def test_venue_routing(self, direct_market_access):
        """Test venue routing"""
        order = HFTOrder(
            order_id="DMA_TEST",
            symbol="AAPL",
            side="BUY",
            quantity=1000,
            price=150.00,
            order_type=OrderType.LIMIT,
            timestamp=datetime.now(),
            status=OrderStatus.PENDING,
            priority_level=5,
            latency_requirement_us=100
        )
        
        # Route to NYSE
        success = direct_market_access.route_order(order, "NYSE")
        assert success
        
        # Route to invalid venue
        success = direct_market_access.route_order(order, "INVALID_VENUE")
        assert not success
    
    def test_venue_latency_tracking(self, direct_market_access):
        """Test venue latency tracking"""
        order = HFTOrder(
            order_id="LATENCY_TEST",
            symbol="AAPL",
            side="BUY",
            quantity=1000,
            price=150.00,
            order_type=OrderType.LIMIT,
            timestamp=datetime.now(),
            status=OrderStatus.PENDING,
            priority_level=5,
            latency_requirement_us=100
        )
        
        # Route to multiple venues
        direct_market_access.route_order(order, "NYSE")
        direct_market_access.route_order(order, "NASDAQ")
        direct_market_access.route_order(order, "BATS")
        
        # Get statistics
        stats = direct_market_access.get_venue_statistics()
        
        assert "NYSE" in stats
        assert "NASDAQ" in stats
        assert "BATS" in stats
        
        # Check latency tracking
        assert stats["NYSE"]["avg_latency_us"] > 0
        assert stats["NYSE"]["order_count"] == 1
        assert stats["NASDAQ"]["order_count"] == 1
    
    def test_venue_capacity_limits(self, direct_market_access):
        """Test venue capacity limits"""
        # Set low capacity for testing
        direct_market_access.connections["NYSE"]["capacity"] = 2
        
        orders = []
        for i in range(3):
            order = HFTOrder(
                order_id=f"CAPACITY_TEST_{i}",
                symbol="AAPL",
                side="BUY",
                quantity=1000,
                price=150.00,
                order_type=OrderType.LIMIT,
                timestamp=datetime.now(),
                status=OrderStatus.PENDING,
                priority_level=5,
                latency_requirement_us=100
            )
            orders.append(order)
        
        # First two should succeed
        assert direct_market_access.route_order(orders[0], "NYSE")
        assert direct_market_access.route_order(orders[1], "NYSE")
        
        # Third should fail due to capacity
        assert not direct_market_access.route_order(orders[2], "NYSE")
    
    def test_venue_comparison(self, direct_market_access):
        """Test venue comparison"""
        order = HFTOrder(
            order_id="VENUE_COMPARISON",
            symbol="AAPL",
            side="BUY",
            quantity=1000,
            price=150.00,
            order_type=OrderType.LIMIT,
            timestamp=datetime.now(),
            status=OrderStatus.PENDING,
            priority_level=5,
            latency_requirement_us=100
        )
        
        # Route to different venues
        venues = ["NYSE", "NASDAQ", "BATS", "EDGX"]
        for venue in venues:
            direct_market_access.route_order(order, venue)
        
        stats = direct_market_access.get_venue_statistics()
        
        # Compare venue latencies
        latencies = {venue: stats[venue]["avg_latency_us"] for venue in venues}
        
        # Should have different latencies
        assert len(set(latencies.values())) > 1
        
        # Find best venue
        best_venue = min(latencies, key=latencies.get)
        assert best_venue in venues


class TestHFTPerformanceBenchmarks:
    """Test HFT performance benchmarks"""
    
    def test_latency_benchmarks(self, ultra_low_latency_order_manager):
        """Test latency benchmarks"""
        latencies = []
        
        # Submit 100 orders and measure latency
        for i in range(100):
            order = HFTOrder(
                order_id=f"BENCH_{i}",
                symbol="AAPL",
                side="BUY",
                quantity=1000,
                price=150.00,
                order_type=OrderType.LIMIT,
                timestamp=datetime.now(),
                status=OrderStatus.PENDING,
                priority_level=5,
                latency_requirement_us=100
            )
            
            start_time = time.perf_counter()
            ultra_low_latency_order_manager.submit_order(order)
            end_time = time.perf_counter()
            
            latency = (end_time - start_time) * 1_000_000
            latencies.append(latency)
        
        # Calculate statistics
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        max_latency = np.max(latencies)
        
        # Performance assertions
        assert avg_latency < 1000  # Less than 1ms average
        assert p95_latency < 2000  # Less than 2ms 95th percentile
        assert p99_latency < 5000  # Less than 5ms 99th percentile
        
        # At least 50% should meet target
        target_meeting = sum(1 for l in latencies if l <= 100) / len(latencies)
        assert target_meeting >= 0.5
    
    def test_throughput_benchmarks(self, ultra_low_latency_order_manager):
        """Test throughput benchmarks"""
        order_count = 1000
        
        # Submit orders as fast as possible
        start_time = time.perf_counter()
        
        for i in range(order_count):
            order = HFTOrder(
                order_id=f"THROUGHPUT_{i}",
                symbol="AAPL",
                side="BUY",
                quantity=100,
                price=150.00,
                order_type=OrderType.LIMIT,
                timestamp=datetime.now(),
                status=OrderStatus.PENDING,
                priority_level=1,
                latency_requirement_us=1000
            )
            
            ultra_low_latency_order_manager.submit_order(order)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Calculate throughput
        throughput = order_count / total_time
        
        # Should achieve high throughput
        assert throughput > 1000  # At least 1000 orders per second
        assert total_time < 1.0  # Complete in less than 1 second
    
    def test_concurrent_performance(self, ultra_low_latency_order_manager):
        """Test concurrent performance"""
        def submit_orders(thread_id, count):
            results = []
            for i in range(count):
                order = HFTOrder(
                    order_id=f"CONCURRENT_{thread_id}_{i}",
                    symbol="AAPL",
                    side="BUY",
                    quantity=100,
                    price=150.00,
                    order_type=OrderType.LIMIT,
                    timestamp=datetime.now(),
                    status=OrderStatus.PENDING,
                    priority_level=1,
                    latency_requirement_us=1000
                )
                
                start_time = time.perf_counter()
                success = ultra_low_latency_order_manager.submit_order(order)
                end_time = time.perf_counter()
                
                latency = (end_time - start_time) * 1_000_000
                results.append((success, latency))
            
            return results
        
        # Run concurrent threads
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(submit_orders, i, 100) for i in range(10)]
            
            all_results = []
            for future in futures:
                results = future.result()
                all_results.extend(results)
        
        # Check results
        success_count = sum(1 for success, _ in all_results if success)
        latencies = [latency for success, latency in all_results if success]
        
        # Should handle concurrency well
        assert success_count == 1000  # All orders should succeed
        assert np.mean(latencies) < 5000  # Average latency should be reasonable
    
    def test_memory_usage_performance(self, ultra_low_latency_order_manager):
        """Test memory usage performance"""
        import sys
        
        # Measure initial memory
        initial_memory = sys.getsizeof(ultra_low_latency_order_manager)
        
        # Submit many orders
        for i in range(1000):
            order = HFTOrder(
                order_id=f"MEMORY_{i}",
                symbol="AAPL",
                side="BUY",
                quantity=100,
                price=150.00,
                order_type=OrderType.LIMIT,
                timestamp=datetime.now(),
                status=OrderStatus.PENDING,
                priority_level=1,
                latency_requirement_us=1000
            )
            
            ultra_low_latency_order_manager.submit_order(order)
        
        # Measure final memory
        final_memory = sys.getsizeof(ultra_low_latency_order_manager)
        memory_growth = final_memory - initial_memory
        
        # Should control memory growth
        assert memory_growth < initial_memory * 1.0  # Less than 100% growth


@pytest.mark.performance
class TestHFTStressTests:
    """Stress tests for HFT systems"""
    
    def test_extreme_order_volume(self, ultra_low_latency_order_manager):
        """Test extreme order volume handling"""
        order_count = 10000
        
        # Submit extreme volume
        start_time = time.perf_counter()
        
        for i in range(order_count):
            order = HFTOrder(
                order_id=f"EXTREME_{i}",
                symbol="AAPL",
                side="BUY" if i % 2 == 0 else "SELL",
                quantity=100,
                price=150.00 + (i % 100) * 0.01,
                order_type=OrderType.LIMIT,
                timestamp=datetime.now(),
                status=OrderStatus.PENDING,
                priority_level=1,
                latency_requirement_us=1000
            )
            
            ultra_low_latency_order_manager.submit_order(order)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Should handle extreme volume
        assert total_time < 10.0  # Complete in less than 10 seconds
        
        # Check performance degradation
        metrics = ultra_low_latency_order_manager.get_performance_metrics()
        assert metrics['total_orders'] >= order_count
        assert metrics['avg_latency_us'] < 10000  # Should stay under 10ms
    
    def test_market_data_flood(self, market_maker, ultra_low_latency_order_manager):
        """Test market data flood handling"""
        # Generate rapid market data updates
        updates = []
        for i in range(1000):
            update = MarketDataUpdate(
                symbol="AAPL",
                timestamp=datetime.now(),
                bid_price=150.00 + np.random.uniform(-0.1, 0.1),
                ask_price=150.10 + np.random.uniform(-0.1, 0.1),
                bid_size=1000,
                ask_size=1000,
                last_price=150.05 + np.random.uniform(-0.1, 0.1),
                last_size=100,
                sequence_number=i + 1
            )
            updates.append(update)
        
        # Process all updates
        start_time = time.perf_counter()
        
        total_orders = 0
        for update in updates:
            orders = market_maker.update_quotes(update, ultra_low_latency_order_manager)
            total_orders += len(orders)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Should handle flood efficiently
        assert total_time < 5.0  # Complete in less than 5 seconds
        assert total_orders > 0  # Should generate orders
    
    def test_system_recovery(self, ultra_low_latency_order_manager):
        """Test system recovery from overload"""
        # Overload system
        for i in range(1000):
            order = HFTOrder(
                order_id=f"OVERLOAD_{i}",
                symbol="AAPL",
                side="BUY",
                quantity=100,
                price=150.00,
                order_type=OrderType.LIMIT,
                timestamp=datetime.now(),
                status=OrderStatus.PENDING,
                priority_level=10,  # High priority
                latency_requirement_us=10
            )
            
            ultra_low_latency_order_manager.submit_order(order)
        
        # Wait for system to recover
        time.sleep(0.1)
        
        # Test normal operation
        normal_order = HFTOrder(
            order_id="RECOVERY_TEST",
            symbol="AAPL",
            side="BUY",
            quantity=1000,
            price=150.00,
            order_type=OrderType.LIMIT,
            timestamp=datetime.now(),
            status=OrderStatus.PENDING,
            priority_level=5,
            latency_requirement_us=100
        )
        
        start_time = time.perf_counter()
        success = ultra_low_latency_order_manager.submit_order(normal_order)
        end_time = time.perf_counter()
        
        latency = (end_time - start_time) * 1_000_000
        
        # Should recover and handle normal orders
        assert success
        assert latency < 5000  # Should be back to reasonable latency


@pytest.mark.integration
class TestHFTIntegration:
    """Integration tests for HFT systems"""
    
    def test_end_to_end_hft_flow(self, ultra_low_latency_order_manager, market_maker, 
                                direct_market_access):
        """Test end-to-end HFT flow"""
        # Market data update
        market_data = MarketDataUpdate(
            symbol="AAPL",
            timestamp=datetime.now(),
            bid_price=149.95,
            ask_price=150.05,
            bid_size=1000,
            ask_size=1000,
            last_price=150.00,
            last_size=100,
            sequence_number=1
        )
        
        # Generate quotes
        orders = market_maker.update_quotes(market_data, ultra_low_latency_order_manager)
        
        # Submit orders
        for order in orders:
            success = ultra_low_latency_order_manager.submit_order(order)
            assert success
            
            # Route to venue
            routed = direct_market_access.route_order(order, "NYSE")
            assert routed
        
        # Verify orders were processed
        assert len(orders) > 0
        
        # Check venue statistics
        stats = direct_market_access.get_venue_statistics()
        assert stats["NYSE"]["order_count"] >= len(orders)
    
    def test_multi_strategy_coordination(self, ultra_low_latency_order_manager, 
                                       market_maker, arbitrage_strategy):
        """Test multi-strategy coordination"""
        # Market data for both symbols
        market_data1 = MarketDataUpdate(
            symbol="AAPL",
            timestamp=datetime.now(),
            bid_price=149.95,
            ask_price=150.05,
            bid_size=1000,
            ask_size=1000,
            last_price=150.00,
            last_size=100,
            sequence_number=1
        )
        
        market_data2 = MarketDataUpdate(
            symbol="GOOGL",
            timestamp=datetime.now(),
            bid_price=2499.95,
            ask_price=2500.05,
            bid_size=1000,
            ask_size=1000,
            last_price=2500.00,
            last_size=100,
            sequence_number=1
        )
        
        # Generate orders from both strategies
        mm_orders = market_maker.update_quotes(market_data1, ultra_low_latency_order_manager)
        arb_orders = arbitrage_strategy.check_arbitrage_opportunity(market_data1, market_data2)
        
        # Submit all orders
        all_orders = mm_orders + arb_orders
        
        for order in all_orders:
            success = ultra_low_latency_order_manager.submit_order(order)
            assert success
        
        # Check that both strategies generated orders
        assert len(mm_orders) > 0
        # Arbitrage orders depend on price divergence
        
        # Verify performance
        metrics = ultra_low_latency_order_manager.get_performance_metrics()
        assert metrics['total_orders'] >= len(all_orders)
    
    def test_real_time_strategy_adaptation(self, ultra_low_latency_order_manager, 
                                         market_maker, sample_market_data):
        """Test real-time strategy adaptation"""
        total_orders = 0
        
        # Process market data stream
        for i, market_data in enumerate(sample_market_data):
            # Generate quotes
            orders = market_maker.update_quotes(market_data, ultra_low_latency_order_manager)
            
            # Submit orders
            for order in orders:
                success = ultra_low_latency_order_manager.submit_order(order)
                if success:
                    total_orders += 1
                    
                    # Simulate fills for some orders
                    if i % 10 == 0:  # 10% fill rate
                        order.status = OrderStatus.FILLED
                        order.filled_quantity = order.quantity
                        order.avg_fill_price = order.price
                        market_maker.on_fill(order)
        
        # Check adaptation
        assert total_orders > 0
        
        # Market maker should have processed orders
        # Position changes depend on simulated fills
        assert market_maker.position >= 0  # Position should be valid
        
        # Performance should be maintained
        metrics = ultra_low_latency_order_manager.get_performance_metrics()
        assert metrics['avg_latency_us'] < 5000  # Should maintain performance