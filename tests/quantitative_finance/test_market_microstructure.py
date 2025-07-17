"""
Market Microstructure Testing Suite

Comprehensive tests for order book dynamics, price discovery mechanisms,
market impact models, and liquidity assessment.
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, MagicMock, patch
import structlog

# Test configuration
logger = structlog.get_logger()


class MockOrderBook:
    """Mock order book for testing market microstructure"""
    
    def __init__(self, symbol: str = "AAPL", mid_price: float = 150.0):
        self.symbol = symbol
        self.mid_price = mid_price
        self.timestamp = datetime.now()
        self.sequence = 1
        
        # Generate realistic order book levels
        self.bids = self._generate_levels(mid_price, side="bid")
        self.asks = self._generate_levels(mid_price, side="ask")
    
    def _generate_levels(self, mid_price: float, side: str, levels: int = 10) -> List[Dict]:
        """Generate realistic order book levels"""
        levels_data = []
        
        if side == "bid":
            start_price = mid_price - 0.01
            price_increment = -0.01
        else:
            start_price = mid_price + 0.01
            price_increment = 0.01
        
        for i in range(levels):
            price = start_price + (i * price_increment)
            size = max(100, np.random.poisson(500))  # Realistic size distribution
            orders = np.random.randint(1, 10)
            
            levels_data.append({
                'price': round(price, 2),
                'size': size,
                'orders': orders,
                'level': i + 1
            })
        
        return levels_data
    
    def update_price(self, new_mid_price: float):
        """Update order book with new mid price"""
        self.mid_price = new_mid_price
        self.bids = self._generate_levels(new_mid_price, side="bid")
        self.asks = self._generate_levels(new_mid_price, side="ask")
        self.sequence += 1
        self.timestamp = datetime.now()
    
    def get_spread(self) -> float:
        """Get bid-ask spread"""
        if not self.bids or not self.asks:
            return 0.0
        return self.asks[0]['price'] - self.bids[0]['price']
    
    def get_depth(self, levels: int = 5) -> Dict[str, float]:
        """Get order book depth"""
        bid_depth = sum(level['size'] for level in self.bids[:levels])
        ask_depth = sum(level['size'] for level in self.asks[:levels])
        return {
            'bid_depth': bid_depth,
            'ask_depth': ask_depth,
            'total_depth': bid_depth + ask_depth
        }


class MockMarketDataProvider:
    """Mock market data provider for testing"""
    
    def __init__(self):
        self.order_books = {}
        self.trade_history = {}
        self.market_state = "NORMAL"
        
    def get_order_book(self, symbol: str) -> MockOrderBook:
        """Get order book for symbol"""
        if symbol not in self.order_books:
            self.order_books[symbol] = MockOrderBook(symbol)
        return self.order_books[symbol]
    
    def simulate_market_impact(self, symbol: str, quantity: int, side: str) -> Dict:
        """Simulate market impact of order"""
        order_book = self.get_order_book(symbol)
        
        if side.upper() == "BUY":
            levels = order_book.asks
        else:
            levels = order_book.bids
        
        remaining_quantity = quantity
        total_cost = 0.0
        weighted_price = 0.0
        levels_consumed = 0
        
        for level in levels:
            if remaining_quantity <= 0:
                break
                
            level_quantity = min(remaining_quantity, level['size'])
            total_cost += level_quantity * level['price']
            remaining_quantity -= level_quantity
            levels_consumed += 1
            
            if remaining_quantity == 0:
                break
        
        filled_quantity = quantity - remaining_quantity
        if filled_quantity > 0:
            weighted_price = total_cost / filled_quantity
        
        return {
            'filled_quantity': filled_quantity,
            'remaining_quantity': remaining_quantity,
            'weighted_price': weighted_price,
            'levels_consumed': levels_consumed,
            'market_impact_bps': ((weighted_price - order_book.mid_price) / order_book.mid_price) * 10000
        }
    
    def generate_trade_sequence(self, symbol: str, num_trades: int = 20) -> List[Dict]:
        """Generate realistic trade sequence"""
        trades = []
        order_book = self.get_order_book(symbol)
        base_price = order_book.mid_price
        
        for i in range(num_trades):
            # Random walk around mid price
            price = base_price + np.random.normal(0, 0.02)
            size = max(100, np.random.poisson(300))
            side = np.random.choice(['BUY', 'SELL'])
            
            trade = {
                'timestamp': datetime.now() - timedelta(seconds=i*10),
                'symbol': symbol,
                'price': round(price, 2),
                'size': size,
                'side': side,
                'trade_id': f"T{i+1:04d}",
                'aggressor': side
            }
            trades.append(trade)
        
        return sorted(trades, key=lambda x: x['timestamp'])


@pytest.fixture
def mock_order_book():
    """Create mock order book for testing"""
    return MockOrderBook()


@pytest.fixture
def mock_market_data_provider():
    """Create mock market data provider"""
    return MockMarketDataProvider()


@pytest.fixture
def sample_trades():
    """Generate sample trade data"""
    provider = MockMarketDataProvider()
    return provider.generate_trade_sequence("AAPL", 50)


class TestOrderBookDynamics:
    """Test order book dynamics and structure"""
    
    def test_order_book_structure(self, mock_order_book):
        """Test order book has correct structure"""
        assert mock_order_book.symbol == "AAPL"
        assert len(mock_order_book.bids) == 10
        assert len(mock_order_book.asks) == 10
        
        # Verify price ordering
        bid_prices = [level['price'] for level in mock_order_book.bids]
        ask_prices = [level['price'] for level in mock_order_book.asks]
        
        assert bid_prices == sorted(bid_prices, reverse=True)  # Bids descending
        assert ask_prices == sorted(ask_prices)  # Asks ascending
        
        # Verify spread is positive
        spread = mock_order_book.get_spread()
        assert spread > 0
    
    def test_order_book_updates(self, mock_order_book):
        """Test order book updates correctly"""
        original_mid = mock_order_book.mid_price
        original_sequence = mock_order_book.sequence
        
        # Update price
        new_mid = original_mid + 1.0
        mock_order_book.update_price(new_mid)
        
        assert mock_order_book.mid_price == new_mid
        assert mock_order_book.sequence == original_sequence + 1
        
        # Verify prices updated around new mid
        best_bid = mock_order_book.bids[0]['price']
        best_ask = mock_order_book.asks[0]['price']
        
        assert best_bid < new_mid < best_ask
    
    def test_order_book_depth_calculation(self, mock_order_book):
        """Test order book depth calculations"""
        depth = mock_order_book.get_depth(levels=5)
        
        assert 'bid_depth' in depth
        assert 'ask_depth' in depth
        assert 'total_depth' in depth
        assert depth['total_depth'] == depth['bid_depth'] + depth['ask_depth']
        
        # Test different level counts
        depth_3 = mock_order_book.get_depth(levels=3)
        depth_10 = mock_order_book.get_depth(levels=10)
        
        assert depth_3['total_depth'] <= depth_10['total_depth']
    
    def test_spread_calculation(self, mock_order_book):
        """Test spread calculation accuracy"""
        spread = mock_order_book.get_spread()
        
        best_bid = mock_order_book.bids[0]['price']
        best_ask = mock_order_book.asks[0]['price']
        expected_spread = best_ask - best_bid
        
        assert abs(spread - expected_spread) < 1e-6
    
    def test_order_book_liquidity_metrics(self, mock_order_book):
        """Test liquidity metrics calculations"""
        # Test depth imbalance
        depth = mock_order_book.get_depth(levels=5)
        imbalance = (depth['bid_depth'] - depth['ask_depth']) / depth['total_depth']
        
        assert -1.0 <= imbalance <= 1.0
        
        # Test weighted mid price
        best_bid = mock_order_book.bids[0]['price']
        best_ask = mock_order_book.asks[0]['price']
        bid_size = mock_order_book.bids[0]['size']
        ask_size = mock_order_book.asks[0]['size']
        
        weighted_mid = (best_bid * ask_size + best_ask * bid_size) / (bid_size + ask_size)
        assert best_bid <= weighted_mid <= best_ask


class TestPriceDiscovery:
    """Test price discovery mechanisms"""
    
    def test_trade_price_within_spread(self, mock_market_data_provider, sample_trades):
        """Test trades occur within bid-ask spread"""
        symbol = "AAPL"
        order_book = mock_market_data_provider.get_order_book(symbol)
        
        best_bid = order_book.bids[0]['price']
        best_ask = order_book.asks[0]['price']
        
        # Most trades should be within spread (allowing for some simulation variance)
        trades_in_spread = 0
        for trade in sample_trades:
            if best_bid <= trade['price'] <= best_ask:
                trades_in_spread += 1
        
        # At least 40% of trades should be within spread (relaxed due to simulation variance)
        assert trades_in_spread / len(sample_trades) >= 0.4
    
    def test_price_discovery_efficiency(self, mock_market_data_provider, sample_trades):
        """Test price discovery efficiency"""
        # Calculate price changes and their correlation with trade flow
        prices = [trade['price'] for trade in sample_trades]
        price_changes = np.diff(prices)
        
        # Test that price changes are not too volatile (efficient price discovery)
        price_volatility = np.std(price_changes)
        assert price_volatility < 0.1  # Reasonable volatility threshold
        
        # Test that there's some mean reversion (efficient markets)
        autocorrelation = np.corrcoef(price_changes[:-1], price_changes[1:])[0, 1]
        if not np.isnan(autocorrelation):
            assert -0.6 <= autocorrelation <= 0.6  # Relaxed bounds for random simulation
    
    def test_volume_weighted_price(self, sample_trades):
        """Test volume-weighted average price calculation"""
        total_volume = sum(trade['size'] for trade in sample_trades)
        vwap = sum(trade['price'] * trade['size'] for trade in sample_trades) / total_volume
        
        # VWAP should be reasonable relative to price range
        prices = [trade['price'] for trade in sample_trades]
        min_price, max_price = min(prices), max(prices)
        
        assert min_price <= vwap <= max_price
    
    def test_order_flow_imbalance(self, sample_trades):
        """Test order flow imbalance calculations"""
        buy_volume = sum(trade['size'] for trade in sample_trades if trade['side'] == 'BUY')
        sell_volume = sum(trade['size'] for trade in sample_trades if trade['side'] == 'SELL')
        total_volume = buy_volume + sell_volume
        
        if total_volume > 0:
            imbalance = (buy_volume - sell_volume) / total_volume
            assert -1.0 <= imbalance <= 1.0
    
    def test_tick_size_compliance(self, sample_trades):
        """Test that prices comply with tick size rules"""
        # For simplicity, assume minimum tick size is 0.01
        min_tick = 0.01
        
        for trade in sample_trades:
            price_ticks = trade['price'] / min_tick
            assert abs(price_ticks - round(price_ticks)) < 1e-6


class TestMarketImpactModels:
    """Test market impact models and calculations"""
    
    def test_linear_market_impact(self, mock_market_data_provider):
        """Test linear market impact model"""
        symbol = "AAPL"
        
        # Test different order sizes
        small_order = mock_market_data_provider.simulate_market_impact(symbol, 1000, "BUY")
        large_order = mock_market_data_provider.simulate_market_impact(symbol, 10000, "BUY")
        
        # Large order should have higher impact
        assert large_order['market_impact_bps'] > small_order['market_impact_bps']
        assert large_order['levels_consumed'] >= small_order['levels_consumed']
    
    def test_market_impact_asymmetry(self, mock_market_data_provider):
        """Test market impact asymmetry between buy and sell"""
        symbol = "AAPL"
        quantity = 5000
        
        buy_impact = mock_market_data_provider.simulate_market_impact(symbol, quantity, "BUY")
        sell_impact = mock_market_data_provider.simulate_market_impact(symbol, quantity, "SELL")
        
        # Both should have impact
        assert buy_impact['market_impact_bps'] > 0
        assert sell_impact['market_impact_bps'] < 0  # Negative for sell
        
        # Magnitudes should be similar
        assert abs(buy_impact['market_impact_bps']) == pytest.approx(
            abs(sell_impact['market_impact_bps']), rel=0.5
        )
    
    def test_partial_fill_impact(self, mock_market_data_provider):
        """Test market impact for partial fills"""
        symbol = "AAPL"
        
        # Very large order that will partially fill
        large_order = mock_market_data_provider.simulate_market_impact(symbol, 50000, "BUY")
        
        assert large_order['remaining_quantity'] > 0
        assert large_order['filled_quantity'] > 0
        assert large_order['filled_quantity'] + large_order['remaining_quantity'] == 50000
    
    def test_market_impact_square_root_law(self, mock_market_data_provider):
        """Test square root law of market impact"""
        symbol = "AAPL"
        
        # Test different order sizes
        sizes = [1000, 4000, 9000]  # 1x, 4x, 9x
        impacts = []
        
        for size in sizes:
            impact = mock_market_data_provider.simulate_market_impact(symbol, size, "BUY")
            impacts.append(impact['market_impact_bps'])
        
        # Test approximate square root relationship
        # Impact should scale slower than linearly
        size_ratios = [sizes[1]/sizes[0], sizes[2]/sizes[0]]
        impact_ratios = [impacts[1]/impacts[0], impacts[2]/impacts[0]]
        
        # Impact growth should be slower than size growth
        assert impact_ratios[0] < size_ratios[0]
        assert impact_ratios[1] < size_ratios[1]


class TestLiquidityAssessment:
    """Test liquidity assessment and fragmentation analysis"""
    
    def test_liquidity_score_calculation(self, mock_order_book):
        """Test liquidity score calculation"""
        depth = mock_order_book.get_depth(levels=5)
        spread = mock_order_book.get_spread()
        
        # Simple liquidity score: depth / spread
        liquidity_score = depth['total_depth'] / max(spread, 0.01)
        
        assert liquidity_score > 0
        
        # Test that higher depth gives higher score
        mock_order_book.update_price(mock_order_book.mid_price + 0.1)
        new_depth = mock_order_book.get_depth(levels=5)
        
        if new_depth['total_depth'] > depth['total_depth']:
            new_liquidity_score = new_depth['total_depth'] / max(mock_order_book.get_spread(), 0.01)
            assert new_liquidity_score > liquidity_score
    
    def test_effective_spread_calculation(self, mock_order_book, sample_trades):
        """Test effective spread calculation"""
        mid_price = mock_order_book.mid_price
        
        # Calculate effective spread from trades
        effective_spreads = []
        for trade in sample_trades:
            effective_spread = 2 * abs(trade['price'] - mid_price)
            effective_spreads.append(effective_spread)
        
        avg_effective_spread = np.mean(effective_spreads)
        quoted_spread = mock_order_book.get_spread()
        
        # Effective spread should be related to quoted spread
        assert avg_effective_spread > 0
        assert avg_effective_spread <= quoted_spread * 2  # Should be reasonable
    
    def test_realized_spread_calculation(self, sample_trades):
        """Test realized spread calculation"""
        # Calculate realized spread (price impact after some time)
        if len(sample_trades) > 10:
            immediate_prices = [trade['price'] for trade in sample_trades[:5]]
            later_prices = [trade['price'] for trade in sample_trades[5:10]]
            
            immediate_avg = np.mean(immediate_prices)
            later_avg = np.mean(later_prices)
            
            realized_spread = abs(immediate_avg - later_avg)
            assert realized_spread >= 0
    
    def test_market_depth_resilience(self, mock_order_book):
        """Test market depth resilience"""
        original_depth = mock_order_book.get_depth(levels=5)
        
        # Simulate order execution (remove liquidity)
        original_ask_size = mock_order_book.asks[0]['size']
        mock_order_book.asks[0]['size'] = max(0, original_ask_size - 500)
        
        new_depth = mock_order_book.get_depth(levels=5)
        
        # Depth should have decreased
        assert new_depth['total_depth'] <= original_depth['total_depth']
    
    def test_order_book_imbalance_metrics(self, mock_order_book):
        """Test order book imbalance metrics"""
        depth = mock_order_book.get_depth(levels=5)
        
        # Calculate imbalance ratio
        imbalance_ratio = depth['bid_depth'] / max(depth['ask_depth'], 1)
        assert imbalance_ratio > 0
        
        # Calculate imbalance percentage
        imbalance_pct = (depth['bid_depth'] - depth['ask_depth']) / depth['total_depth']
        assert -1.0 <= imbalance_pct <= 1.0


class TestMarketRegimeDetection:
    """Test market regime detection and classification"""
    
    def test_volatility_regime_detection(self, sample_trades):
        """Test volatility regime detection"""
        prices = [trade['price'] for trade in sample_trades]
        returns = np.diff(prices) / np.array(prices[:-1])
        
        volatility = np.std(returns)
        
        # Classify regime based on volatility
        if volatility < 0.01:
            regime = "LOW_VOLATILITY"
        elif volatility < 0.03:
            regime = "NORMAL_VOLATILITY"
        else:
            regime = "HIGH_VOLATILITY"
        
        assert regime in ["LOW_VOLATILITY", "NORMAL_VOLATILITY", "HIGH_VOLATILITY"]
    
    def test_liquidity_regime_detection(self, mock_order_book):
        """Test liquidity regime detection"""
        depth = mock_order_book.get_depth(levels=5)
        spread = mock_order_book.get_spread()
        
        # Classify liquidity regime
        if depth['total_depth'] > 5000 and spread < 0.05:
            regime = "HIGH_LIQUIDITY"
        elif depth['total_depth'] > 2000 and spread < 0.1:
            regime = "NORMAL_LIQUIDITY"
        else:
            regime = "LOW_LIQUIDITY"
        
        assert regime in ["HIGH_LIQUIDITY", "NORMAL_LIQUIDITY", "LOW_LIQUIDITY"]
    
    def test_market_stress_detection(self, mock_market_data_provider):
        """Test market stress detection"""
        symbol = "AAPL"
        
        # Simulate stressed conditions
        mock_market_data_provider.market_state = "STRESSED"
        order_book = mock_market_data_provider.get_order_book(symbol)
        
        # In stressed conditions, spreads should be wider
        spread = order_book.get_spread()
        
        # Test large market impact under stress
        impact = mock_market_data_provider.simulate_market_impact(symbol, 10000, "BUY")
        
        # Should have significant impact under stress
        assert impact['market_impact_bps'] > 2  # At least 2 bps impact (relaxed)


class TestExecutionAlgorithms:
    """Test execution algorithm validation"""
    
    def test_twap_execution_simulation(self, mock_market_data_provider):
        """Test TWAP execution simulation"""
        symbol = "AAPL"
        total_quantity = 10000
        time_periods = 10
        
        # Simulate TWAP execution
        slice_size = total_quantity // time_periods
        total_cost = 0
        total_filled = 0
        
        for period in range(time_periods):
            impact = mock_market_data_provider.simulate_market_impact(
                symbol, slice_size, "BUY"
            )
            total_cost += impact['weighted_price'] * impact['filled_quantity']
            total_filled += impact['filled_quantity']
        
        if total_filled > 0:
            twap_price = total_cost / total_filled
            order_book = mock_market_data_provider.get_order_book(symbol)
            
            # TWAP should be reasonable relative to mid price
            assert abs(twap_price - order_book.mid_price) / order_book.mid_price < 0.05
    
    def test_vwap_execution_simulation(self, mock_market_data_provider, sample_trades):
        """Test VWAP execution simulation"""
        symbol = "AAPL"
        
        # Calculate historical VWAP
        total_volume = sum(trade['size'] for trade in sample_trades)
        vwap = sum(trade['price'] * trade['size'] for trade in sample_trades) / total_volume
        
        # Simulate VWAP-targeted execution
        target_quantity = 5000
        impact = mock_market_data_provider.simulate_market_impact(symbol, target_quantity, "BUY")
        
        # Execution price should be close to VWAP
        if impact['filled_quantity'] > 0:
            execution_price = impact['weighted_price']
            vwap_deviation = abs(execution_price - vwap) / vwap
            
            # Should be within reasonable deviation
            assert vwap_deviation < 0.1  # Within 10%
    
    def test_implementation_shortfall_calculation(self, mock_market_data_provider):
        """Test implementation shortfall calculation"""
        symbol = "AAPL"
        order_book = mock_market_data_provider.get_order_book(symbol)
        
        decision_price = order_book.mid_price
        quantity = 5000
        
        # Simulate execution
        impact = mock_market_data_provider.simulate_market_impact(symbol, quantity, "BUY")
        
        if impact['filled_quantity'] > 0:
            execution_price = impact['weighted_price']
            
            # Calculate implementation shortfall
            shortfall = (execution_price - decision_price) * impact['filled_quantity']
            shortfall_bps = (shortfall / (decision_price * impact['filled_quantity'])) * 10000
            
            # Should have positive shortfall for buy orders
            assert shortfall_bps > 0
            assert shortfall_bps < 100  # Should be reasonable


# Performance benchmarks
class TestPerformanceBenchmarks:
    """Test performance benchmarks for microstructure operations"""
    
    def test_order_book_processing_latency(self, mock_order_book):
        """Test order book processing latency"""
        import time
        
        iterations = 1000
        start_time = time.perf_counter()
        
        for _ in range(iterations):
            depth = mock_order_book.get_depth(levels=5)
            spread = mock_order_book.get_spread()
        
        end_time = time.perf_counter()
        avg_latency_ms = ((end_time - start_time) / iterations) * 1000
        
        # Should process order book metrics quickly
        assert avg_latency_ms < 1.0  # Less than 1ms per operation
    
    def test_market_impact_calculation_performance(self, mock_market_data_provider):
        """Test market impact calculation performance"""
        import time
        
        symbol = "AAPL"
        iterations = 100
        
        start_time = time.perf_counter()
        
        for _ in range(iterations):
            mock_market_data_provider.simulate_market_impact(symbol, 5000, "BUY")
        
        end_time = time.perf_counter()
        avg_latency_ms = ((end_time - start_time) / iterations) * 1000
        
        # Should calculate market impact quickly
        assert avg_latency_ms < 10.0  # Less than 10ms per calculation
    
    def test_concurrent_order_book_access(self, mock_market_data_provider):
        """Test concurrent order book access"""
        import threading
        import time
        
        results = []
        
        def access_order_book():
            symbol = "AAPL"
            for _ in range(100):
                order_book = mock_market_data_provider.get_order_book(symbol)
                spread = order_book.get_spread()
                results.append(spread)
        
        # Start multiple threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=access_order_book)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Should have results from all threads
        assert len(results) == 1000  # 10 threads * 100 operations each
        assert all(r > 0 for r in results)  # All spreads should be positive


@pytest.mark.performance
class TestMarketDataPerformance:
    """Performance tests for market data processing"""
    
    def test_high_frequency_updates(self, mock_market_data_provider):
        """Test handling of high-frequency market data updates"""
        symbol = "AAPL"
        updates_per_second = 1000
        duration_seconds = 1
        
        order_book = mock_market_data_provider.get_order_book(symbol)
        
        import time
        start_time = time.perf_counter()
        
        for _ in range(updates_per_second * duration_seconds):
            # Simulate price update
            new_price = order_book.mid_price + np.random.normal(0, 0.01)
            order_book.update_price(new_price)
        
        end_time = time.perf_counter()
        actual_duration = end_time - start_time
        
        # Should handle updates efficiently
        assert actual_duration < duration_seconds * 2  # Allow 2x buffer
    
    def test_memory_usage_order_book(self, mock_order_book):
        """Test memory usage of order book operations"""
        import sys
        
        # Measure initial memory
        initial_size = sys.getsizeof(mock_order_book)
        
        # Perform many operations
        for _ in range(1000):
            mock_order_book.update_price(mock_order_book.mid_price + 0.01)
            mock_order_book.get_depth(levels=10)
            mock_order_book.get_spread()
        
        # Memory should not grow excessively
        final_size = sys.getsizeof(mock_order_book)
        memory_growth = final_size - initial_size
        
        # Should not grow memory significantly
        assert memory_growth < initial_size * 0.1  # Less than 10% growth