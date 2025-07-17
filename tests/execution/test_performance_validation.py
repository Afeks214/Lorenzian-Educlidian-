"""
Performance Validation Tests

Critical performance tests to validate <500μs order placement latency
and >99.8% fill rate requirements for the execution system.
"""

import pytest
import asyncio
import time
import statistics
from typing import List, Dict, Any
from datetime import datetime, timedelta
import concurrent.futures

from src.execution.order_management.order_manager import OrderManager, OrderManagerConfig
from src.execution.order_management.order_types import OrderRequest, OrderSide, OrderType
from src.execution.order_management.order_validator import OrderValidator, ValidationConfig
from src.execution.routing.smart_router import SmartOrderRouter, SmartRouterConfig
from src.execution.routing.venue_manager import VenueManager, VenueConfig, VenueType
from src.execution.brokers.broker_factory import BrokerFactory, BrokerType
from src.core.events import EventBus
import structlog

logger = structlog.get_logger()


class PerformanceTestSuite:
    """
    Comprehensive performance test suite for execution system.
    
    Validates critical performance requirements:
    - <500μs order placement latency
    - >99.8% fill rate achievement
    - High-frequency order processing
    """
    
    def __init__(self):
        self.event_bus = EventBus()
        self.setup_execution_system()
        
        # Performance tracking
        self.latency_measurements: List[float] = []
        self.fill_rate_measurements: List[float] = []
        self.throughput_measurements: List[float] = []
        
    def setup_execution_system(self):
        """Setup complete execution system for testing"""
        
        # Create venue configurations
        venue_configs = [
            VenueConfig(
                venue_id="VENUE_1",
                name="Test Venue 1",
                venue_type=VenueType.EXCHANGE,
                api_endpoint="http://test-venue-1.com",
                expected_latency_ms=5.0,
                cost_per_share=0.001,
                supports_market_orders=True,
                supports_limit_orders=True
            ),
            VenueConfig(
                venue_id="VENUE_2", 
                name="Test Venue 2",
                venue_type=VenueType.ECN,
                api_endpoint="http://test-venue-2.com",
                expected_latency_ms=3.0,
                cost_per_share=0.0008,
                supports_market_orders=True,
                supports_limit_orders=True
            )
        ]
        
        # Create venue manager
        self.venue_manager = VenueManager(venue_configs)
        
        # Create order validator
        validation_config = ValidationConfig(
            max_order_value={'default': 10000000},
            daily_order_limit=50000
        )
        self.order_validator = OrderValidator(validation_config)
        
        # Create smart router
        router_config = SmartRouterConfig(
            max_routing_latency_us=50.0,  # Very tight requirement
            target_fill_rate=0.998
        )
        self.smart_router = SmartOrderRouter(
            router_config, 
            self.venue_manager
        )
        
        # Create order manager
        order_config = OrderManagerConfig(
            enable_fast_path=True,
            worker_threads=8,
            enable_execution_tracking=True
        )
        self.order_manager = OrderManager(
            order_config,
            self.event_bus,
            self.smart_router,
            self.order_validator
        )
        
        # Create simulated broker
        self.broker = BrokerFactory.create_simulated_client({
            'simulation_latency_ms': 2,  # Very fast simulation
            'fill_probability': 0.999,   # High fill rate
            'slippage_bps': 0.5          # Low slippage
        })


@pytest.mark.performance
class TestOrderPlacementLatency:
    """Test order placement latency requirements"""
    
    @pytest.fixture
    def performance_suite(self):
        return PerformanceTestSuite()
    
    @pytest.mark.asyncio
    async def test_single_order_latency(self, performance_suite):
        """Test single order placement latency <500μs"""
        
        # Connect to broker
        await performance_suite.broker.connect()
        
        # Create test order
        order_request = OrderRequest(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET
        )
        
        # Measure order placement latency
        start_time = time.perf_counter()
        
        order_id = await performance_suite.order_manager.submit_order(order_request)
        
        end_time = time.perf_counter()
        latency_us = (end_time - start_time) * 1_000_000
        
        # Verify latency requirement
        assert latency_us < 500, f"Order placement latency {latency_us:.1f}μs exceeds 500μs target"
        
        # Verify order was created
        assert order_id is not None
        
        # Log performance
        logger.info(f"Single order latency: {latency_us:.1f}μs")
    
    @pytest.mark.asyncio
    async def test_fast_path_latency(self, performance_suite):
        """Test fast path order placement for urgent orders"""
        
        await performance_suite.broker.connect()
        
        # Create urgent order (should use fast path)
        order_request = OrderRequest(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
            priority=OrderPriority.URGENT
        )
        
        # Measure fast path latency
        start_time = time.perf_counter()
        
        order_id = await performance_suite.order_manager.submit_order(order_request)
        
        end_time = time.perf_counter()
        latency_us = (end_time - start_time) * 1_000_000
        
        # Fast path should be even faster
        assert latency_us < 200, f"Fast path latency {latency_us:.1f}μs exceeds 200μs target"
        
        logger.info(f"Fast path latency: {latency_us:.1f}μs")
    
    @pytest.mark.asyncio
    async def test_batch_order_latency_consistency(self, performance_suite):
        """Test latency consistency across multiple orders"""
        
        await performance_suite.broker.connect()
        
        latencies = []
        num_orders = 100
        
        for i in range(num_orders):
            order_request = OrderRequest(
                symbol=f"TEST{i%10}",  # Cycle through 10 symbols
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                quantity=100 + i,
                order_type=OrderType.MARKET
            )
            
            start_time = time.perf_counter()
            order_id = await performance_suite.order_manager.submit_order(order_request)
            end_time = time.perf_counter()
            
            latency_us = (end_time - start_time) * 1_000_000
            latencies.append(latency_us)
            
            assert order_id is not None
        
        # Analyze latency statistics
        avg_latency = statistics.mean(latencies)
        max_latency = max(latencies)
        p95_latency = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
        p99_latency = statistics.quantiles(latencies, n=100)[98]  # 99th percentile
        
        # All orders should meet latency target
        orders_meeting_target = sum(1 for lat in latencies if lat < 500)
        success_rate = orders_meeting_target / num_orders
        
        assert success_rate >= 0.99, f"Only {success_rate:.1%} of orders met latency target"
        assert avg_latency < 300, f"Average latency {avg_latency:.1f}μs too high"
        assert p95_latency < 500, f"95th percentile latency {p95_latency:.1f}μs exceeds target"
        
        logger.info(
            f"Batch latency stats: avg={avg_latency:.1f}μs, "
            f"max={max_latency:.1f}μs, p95={p95_latency:.1f}μs, p99={p99_latency:.1f}μs"
        )
    
    @pytest.mark.asyncio
    async def test_concurrent_order_latency(self, performance_suite):
        """Test latency under concurrent load"""
        
        await performance_suite.broker.connect()
        
        async def submit_order_batch(batch_id: int, orders_per_batch: int):
            """Submit a batch of orders concurrently"""
            batch_latencies = []
            
            for i in range(orders_per_batch):
                order_request = OrderRequest(
                    symbol=f"BATCH{batch_id}",
                    side=OrderSide.BUY,
                    quantity=100,
                    order_type=OrderType.MARKET
                )
                
                start_time = time.perf_counter()
                order_id = await performance_suite.order_manager.submit_order(order_request)
                end_time = time.perf_counter()
                
                latency_us = (end_time - start_time) * 1_000_000
                batch_latencies.append(latency_us)
            
            return batch_latencies
        
        # Run multiple concurrent batches
        num_batches = 10
        orders_per_batch = 20
        
        tasks = [
            submit_order_batch(batch_id, orders_per_batch)
            for batch_id in range(num_batches)
        ]
        
        batch_results = await asyncio.gather(*tasks)
        
        # Flatten all latencies
        all_latencies = [lat for batch in batch_results for lat in batch]
        
        # Analyze concurrent performance
        avg_concurrent_latency = statistics.mean(all_latencies)
        max_concurrent_latency = max(all_latencies)
        
        # Under concurrent load, allow slightly higher latency
        orders_meeting_target = sum(1 for lat in all_latencies if lat < 750)  # 750μs under load
        success_rate = orders_meeting_target / len(all_latencies)
        
        assert success_rate >= 0.95, f"Only {success_rate:.1%} met latency target under load"
        assert avg_concurrent_latency < 500, f"Average concurrent latency {avg_concurrent_latency:.1f}μs too high"
        
        logger.info(
            f"Concurrent latency: avg={avg_concurrent_latency:.1f}μs, "
            f"max={max_concurrent_latency:.1f}μs, success_rate={success_rate:.1%}"
        )


@pytest.mark.performance  
class TestFillRateRequirements:
    """Test fill rate requirements >99.8%"""
    
    @pytest.fixture
    def performance_suite(self):
        return PerformanceTestSuite()
    
    @pytest.mark.asyncio
    async def test_market_order_fill_rate(self, performance_suite):
        """Test fill rate for market orders"""
        
        await performance_suite.broker.connect()
        
        num_orders = 1000
        filled_orders = 0
        
        for i in range(num_orders):
            order_request = OrderRequest(
                symbol="AAPL",
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                quantity=100,
                order_type=OrderType.MARKET
            )
            
            order_id = await performance_suite.order_manager.submit_order(order_request)
            
            # Wait briefly for execution
            await asyncio.sleep(0.01)
            
            # Check if order was filled
            order = performance_suite.order_manager.get_order(order_id)
            if order and order.is_complete:
                filled_orders += 1
        
        fill_rate = filled_orders / num_orders
        
        assert fill_rate >= 0.998, f"Fill rate {fill_rate:.3%} below 99.8% target"
        
        logger.info(f"Market order fill rate: {fill_rate:.3%}")
    
    @pytest.mark.asyncio
    async def test_limit_order_fill_rate(self, performance_suite):
        """Test fill rate for limit orders at reasonable prices"""
        
        await performance_suite.broker.connect()
        
        num_orders = 500
        filled_orders = 0
        
        base_price = 150.0
        
        for i in range(num_orders):
            # Create limit orders near market price (should fill)
            side = OrderSide.BUY if i % 2 == 0 else OrderSide.SELL
            
            if side == OrderSide.BUY:
                limit_price = base_price + 0.10  # Buy above market (should fill)
            else:
                limit_price = base_price - 0.10  # Sell below market (should fill)
            
            order_request = OrderRequest(
                symbol="AAPL",
                side=side,
                quantity=100,
                order_type=OrderType.LIMIT,
                price=limit_price
            )
            
            order_id = await performance_suite.order_manager.submit_order(order_request)
            
            # Wait for execution
            await asyncio.sleep(0.02)
            
            order = performance_suite.order_manager.get_order(order_id)
            if order and order.is_complete:
                filled_orders += 1
        
        fill_rate = filled_orders / num_orders
        
        # Limit orders at good prices should still have high fill rate
        assert fill_rate >= 0.95, f"Limit order fill rate {fill_rate:.3%} too low"
        
        logger.info(f"Limit order fill rate: {fill_rate:.3%}")
    
    @pytest.mark.asyncio
    async def test_mixed_order_type_fill_rate(self, performance_suite):
        """Test fill rate across mixed order types"""
        
        await performance_suite.broker.connect()
        
        order_types = [OrderType.MARKET, OrderType.LIMIT, OrderType.MARKET, OrderType.LIMIT]
        num_cycles = 250  # 1000 total orders
        
        filled_orders = 0
        total_orders = 0
        
        for cycle in range(num_cycles):
            for order_type in order_types:
                total_orders += 1
                
                if order_type == OrderType.MARKET:
                    order_request = OrderRequest(
                        symbol="MIXED_TEST",
                        side=OrderSide.BUY if total_orders % 2 == 0 else OrderSide.SELL,
                        quantity=100,
                        order_type=OrderType.MARKET
                    )
                else:
                    # Aggressive limit order
                    side = OrderSide.BUY if total_orders % 2 == 0 else OrderSide.SELL
                    base_price = 150.0
                    
                    if side == OrderSide.BUY:
                        price = base_price + 0.05
                    else:
                        price = base_price - 0.05
                    
                    order_request = OrderRequest(
                        symbol="MIXED_TEST",
                        side=side,
                        quantity=100,
                        order_type=OrderType.LIMIT,
                        price=price
                    )
                
                order_id = await performance_suite.order_manager.submit_order(order_request)
                
                # Wait for execution
                await asyncio.sleep(0.01)
                
                order = performance_suite.order_manager.get_order(order_id)
                if order and order.is_complete:
                    filled_orders += 1
        
        overall_fill_rate = filled_orders / total_orders
        
        assert overall_fill_rate >= 0.998, f"Overall fill rate {overall_fill_rate:.3%} below target"
        
        logger.info(f"Mixed order fill rate: {overall_fill_rate:.3%}")


@pytest.mark.performance
class TestThroughputRequirements:
    """Test order processing throughput"""
    
    @pytest.fixture
    def performance_suite(self):
        return PerformanceTestSuite()
    
    @pytest.mark.asyncio
    async def test_sequential_throughput(self, performance_suite):
        """Test sequential order processing throughput"""
        
        await performance_suite.broker.connect()
        
        num_orders = 1000
        
        start_time = time.perf_counter()
        
        for i in range(num_orders):
            order_request = OrderRequest(
                symbol=f"THROUGHPUT{i%10}",
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                quantity=100,
                order_type=OrderType.MARKET
            )
            
            await performance_suite.order_manager.submit_order(order_request)
        
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        throughput = num_orders / total_time
        
        # Should handle at least 1000 orders/second
        assert throughput >= 1000, f"Sequential throughput {throughput:.0f} orders/sec too low"
        
        logger.info(f"Sequential throughput: {throughput:.0f} orders/second")
    
    @pytest.mark.asyncio
    async def test_concurrent_throughput(self, performance_suite):
        """Test concurrent order processing throughput"""
        
        await performance_suite.broker.connect()
        
        async def submit_orders_concurrently(order_count: int):
            """Submit orders concurrently"""
            tasks = []
            
            for i in range(order_count):
                order_request = OrderRequest(
                    symbol=f"CONCURRENT{i%20}",
                    side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                    quantity=100,
                    order_type=OrderType.MARKET
                )
                
                task = performance_suite.order_manager.submit_order(order_request)
                tasks.append(task)
            
            await asyncio.gather(*tasks)
        
        num_orders = 2000
        
        start_time = time.perf_counter()
        await submit_orders_concurrently(num_orders)
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        concurrent_throughput = num_orders / total_time
        
        # Concurrent processing should be much faster
        assert concurrent_throughput >= 5000, f"Concurrent throughput {concurrent_throughput:.0f} orders/sec too low"
        
        logger.info(f"Concurrent throughput: {concurrent_throughput:.0f} orders/second")


@pytest.mark.performance
class TestSystemStressTest:
    """Stress test the complete execution system"""
    
    @pytest.fixture
    def performance_suite(self):
        return PerformanceTestSuite()
    
    @pytest.mark.asyncio
    async def test_sustained_load(self, performance_suite):
        """Test system under sustained load"""
        
        await performance_suite.broker.connect()
        
        # Run sustained load for 30 seconds
        duration_seconds = 30
        target_rate = 100  # orders per second
        
        start_time = time.perf_counter()
        end_time = start_time + duration_seconds
        
        order_count = 0
        latencies = []
        
        while time.perf_counter() < end_time:
            batch_start = time.perf_counter()
            
            # Submit batch of orders
            batch_size = 10
            batch_tasks = []
            
            for i in range(batch_size):
                order_request = OrderRequest(
                    symbol=f"STRESS{order_count%5}",
                    side=OrderSide.BUY if order_count % 2 == 0 else OrderSide.SELL,
                    quantity=100,
                    order_type=OrderType.MARKET
                )
                
                order_start = time.perf_counter()
                task = performance_suite.order_manager.submit_order(order_request)
                batch_tasks.append((task, order_start))
                order_count += 1
            
            # Wait for batch completion and measure latencies
            for task, order_start in batch_tasks:
                await task
                order_end = time.perf_counter()
                latency_us = (order_end - order_start) * 1_000_000
                latencies.append(latency_us)
            
            # Rate limiting
            batch_time = time.perf_counter() - batch_start
            target_batch_time = batch_size / target_rate
            
            if batch_time < target_batch_time:
                await asyncio.sleep(target_batch_time - batch_time)
        
        actual_duration = time.perf_counter() - start_time
        actual_rate = order_count / actual_duration
        
        # Analyze sustained performance
        avg_latency = statistics.mean(latencies)
        max_latency = max(latencies)
        p99_latency = statistics.quantiles(latencies, n=100)[98]
        
        # Performance should remain stable under sustained load
        assert avg_latency < 600, f"Average latency {avg_latency:.1f}μs degraded under load"
        assert p99_latency < 1000, f"P99 latency {p99_latency:.1f}μs too high under load"
        assert actual_rate >= target_rate * 0.95, f"Failed to maintain target rate: {actual_rate:.1f}/sec"
        
        logger.info(
            f"Sustained load test: {order_count} orders in {actual_duration:.1f}s "
            f"({actual_rate:.1f} orders/sec), avg latency: {avg_latency:.1f}μs"
        )
    
    @pytest.mark.asyncio 
    async def test_memory_usage_stability(self, performance_suite):
        """Test memory usage remains stable under load"""
        
        import psutil
        import os
        
        await performance_suite.broker.connect()
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Submit many orders to test memory stability
        num_orders = 5000
        
        for i in range(num_orders):
            order_request = OrderRequest(
                symbol=f"MEMORY{i%100}",
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                quantity=100,
                order_type=OrderType.MARKET
            )
            
            await performance_suite.order_manager.submit_order(order_request)
            
            # Check memory every 1000 orders
            if i % 1000 == 999:
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_growth = current_memory - initial_memory
                
                # Memory growth should be reasonable
                assert memory_growth < 200, f"Excessive memory growth: {memory_growth:.1f}MB"
        
        final_memory = process.memory_info().rss / 1024 / 1024
        total_growth = final_memory - initial_memory
        
        logger.info(
            f"Memory usage: initial={initial_memory:.1f}MB, "
            f"final={final_memory:.1f}MB, growth={total_growth:.1f}MB"
        )
        
        # Total memory growth should be reasonable
        assert total_growth < 300, f"Total memory growth too high: {total_growth:.1f}MB"


if __name__ == "__main__":
    # Run performance tests
    pytest.main([
        __file__,
        "-v",
        "-m", "performance",
        "--tb=short"
    ])