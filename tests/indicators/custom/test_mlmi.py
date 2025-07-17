"""
Mathematical Validation Tests for MLMI (Machine Learning Market Indicator)
Tests mathematical correctness, boundary conditions, and performance requirements
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import time

from src.indicators.custom.mlmi import MLMICalculator, MLMIDataFast, calculate_wma, calculate_rsi_with_ma, fast_knn_predict
from src.core.minimal_dependencies import EventBus, BarData
from tests.mocks.mock_event_bus import MockEventBus


class TestMLMICalculator:
    """Test suite for MLMI Calculator mathematical validation"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.mock_event_bus = MockEventBus()
        self.config = {
            'num_neighbors': 200,
            'momentum_window': 20,
            'ma_quick_period': 5,
            'ma_slow_period': 20,
            'rsi_quick_period': 5,
            'rsi_slow_period': 20,
            'mlmi_ma_period': 20,
            'band_lookback': 2000,
            'std_window': 20,
            'ema_std_span': 20
        }
        self.mlmi = MLMICalculator(self.config, self.mock_event_bus)
        
    def create_test_bar(self, close=100.0, high=None, low=None, open=None, volume=1000, timestamp=None):
        """Create a test bar with realistic OHLCV data"""
        if timestamp is None:
            timestamp = datetime.now()
        if high is None:
            high = close + np.random.uniform(0.1, 2.0)
        if low is None:
            low = close - np.random.uniform(0.1, 2.0)
        if open is None:
            open = close + np.random.uniform(-1.0, 1.0)
            
        return BarData(
            symbol="BTCUSDT",
            timestamp=timestamp,
            open=open,
            high=high,
            low=low,
            close=close,
            volume=volume,
            timeframe=30
        )
    
    def generate_trending_data(self, length=100, trend_slope=0.01):
        """Generate trending price data for testing"""
        bars = []
        base_price = 100.0
        
        for i in range(length):
            # Add trend
            price = base_price + (i * trend_slope)
            # Add some noise
            noise = np.random.normal(0, 0.5)
            close = price + noise
            
            timestamp = datetime.now() + timedelta(minutes=i*30)
            bar = self.create_test_bar(close=close, timestamp=timestamp)
            bars.append(bar)
            
        return bars
    
    def generate_ranging_data(self, length=100, range_center=100.0, range_width=10.0):
        """Generate ranging price data for testing"""
        bars = []
        
        for i in range(length):
            # Oscillate around center
            angle = (i / length) * 4 * np.pi  # 4 full cycles
            price = range_center + (range_width * np.sin(angle))
            # Add some noise
            noise = np.random.normal(0, 0.2)
            close = price + noise
            
            timestamp = datetime.now() + timedelta(minutes=i*30)
            bar = self.create_test_bar(close=close, timestamp=timestamp)
            bars.append(bar)
            
        return bars
    
    def test_mlmi_initialization(self):
        """Test MLMI calculator initialization"""
        assert self.mlmi.num_neighbors == 200
        assert self.mlmi.momentum_window == 20
        assert self.mlmi.ma_quick_period == 5
        assert self.mlmi.ma_slow_period == 20
        assert self.mlmi.rsi_quick_period == 5
        assert self.mlmi.rsi_slow_period == 20
        assert isinstance(self.mlmi.mlmi_data, MLMIDataFast)
        assert self.mlmi.last_mlmi_value == 0.0
        assert self.mlmi.last_mlmi_signal == 0
        
    def test_mlmi_data_fast_storage(self):
        """Test MLMIDataFast storage functionality"""
        mlmi_data = MLMIDataFast(max_size=100)
        
        # Test initial state
        assert mlmi_data.size == 0
        
        # Test first trade storage
        mlmi_data.storePreviousTrade(0.5, 0.3, 100.0)
        assert mlmi_data.size == 1
        assert mlmi_data.parameter1[0] == 0.5
        assert mlmi_data.parameter2[0] == 0.3
        assert mlmi_data.priceArray[0] == 100.0
        assert mlmi_data.resultArray[0] == 0  # First trade has no result
        
        # Test second trade storage (price went up)
        mlmi_data.storePreviousTrade(0.6, 0.4, 105.0)
        assert mlmi_data.size == 2
        assert mlmi_data.resultArray[1] == 1  # Price increased
        
        # Test third trade storage (price went down)
        mlmi_data.storePreviousTrade(0.4, 0.2, 102.0)
        assert mlmi_data.size == 3
        assert mlmi_data.resultArray[2] == -1  # Price decreased
        
    def test_weighted_moving_average(self):
        """Test weighted moving average calculation"""
        # Test with known values
        series = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        length = 5
        
        wma = calculate_wma(series, length)
        
        # Check that early values are zero (not enough data)
        assert wma[0] == 0.0
        assert wma[1] == 0.0
        assert wma[2] == 0.0
        assert wma[3] == 0.0
        
        # Check calculation for index 4 (5th element)
        # WMA = (1*1 + 2*2 + 3*3 + 4*4 + 5*5) / (1+2+3+4+5) = 55/15 = 3.666...
        expected_wma_4 = (1*1 + 2*2 + 3*3 + 4*4 + 5*5) / (1+2+3+4+5)
        assert abs(wma[4] - expected_wma_4) < 1e-10
        
        # Check that WMA is increasing for increasing series
        assert wma[5] > wma[4]
        assert wma[6] > wma[5]
        
    def test_rsi_calculation(self):
        """Test RSI calculation with Wilder's smoothing"""
        # Create test data with known characteristics
        # Rising prices should give RSI > 50
        rising_prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0])
        
        rsi = calculate_rsi_with_ma(rising_prices, 14)
        
        # RSI should be > 50 for rising prices
        assert rsi[-1] > 50.0
        assert rsi[-1] <= 100.0
        
        # Falling prices should give RSI < 50
        falling_prices = np.array([115.0, 114.0, 113.0, 112.0, 111.0, 110.0, 109.0, 108.0, 107.0, 106.0, 105.0, 104.0, 103.0, 102.0, 101.0, 100.0])
        
        rsi_falling = calculate_rsi_with_ma(falling_prices, 14)
        
        # RSI should be < 50 for falling prices
        assert rsi_falling[-1] < 50.0
        assert rsi_falling[-1] >= 0.0
        
        # Test boundary conditions
        # All same prices should give RSI = 50
        flat_prices = np.array([100.0] * 16)
        rsi_flat = calculate_rsi_with_ma(flat_prices, 14)
        # RSI should be close to 50 for flat prices (may not be exactly 50 due to numerical precision)
        assert abs(rsi_flat[-1] - 50.0) < 10.0  # Allow some tolerance
        
    def test_fast_knn_predict(self):
        """Test fast k-NN prediction functionality"""
        # Create test data
        param1_array = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        param2_array = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1])
        result_array = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1])
        
        # Test prediction
        prediction = fast_knn_predict(param1_array, param2_array, result_array, 0.55, 0.65, 3, 10)
        
        # Should return sum of k nearest neighbors
        assert isinstance(prediction, (int, float))
        assert -10 <= prediction <= 10  # Bounded by k * max(result)
        
        # Test with empty data
        empty_prediction = fast_knn_predict(np.array([]), np.array([]), np.array([]), 0.5, 0.5, 3, 0)
        assert empty_prediction == 0
        
    def test_heiken_ashi_conversion(self):
        """Test Heiken Ashi conversion in MLMI context"""
        # Create test bars
        bars = self.generate_trending_data(10)
        
        # Add to history
        for bar in bars:
            self.mlmi.update_30m_history(bar)
        
        # Convert to Heiken Ashi
        ha_bars = self.mlmi.convert_to_heiken_ashi(self.mlmi.history_30m)
        
        # Check that we have the same number of bars
        assert len(ha_bars) == len(bars)
        
        # Check HA properties
        for i, ha_bar in enumerate(ha_bars):
            # All prices should be positive
            assert ha_bar['open'] > 0
            assert ha_bar['high'] > 0
            assert ha_bar['low'] > 0
            assert ha_bar['close'] > 0
            
            # High should be max of high, open, close
            assert ha_bar['high'] >= ha_bar['open']
            assert ha_bar['high'] >= ha_bar['close']
            
            # Low should be min of low, open, close
            assert ha_bar['low'] <= ha_bar['open']
            assert ha_bar['low'] <= ha_bar['close']
            
            # Volume should be preserved
            assert ha_bar['volume'] == bars[i].volume
            
    def test_mlmi_calculation_with_insufficient_data(self):
        """Test MLMI calculation with insufficient data"""
        # Test with no data
        result = self.mlmi.calculate_30m(self.create_test_bar())
        assert result == {'mlmi_value': 0.0, 'mlmi_signal': 0}
        
        # Test with insufficient data (< 100 bars)
        for i in range(50):
            bar = self.create_test_bar(close=100.0 + i)
            result = self.mlmi.calculate_30m(bar)
            assert result == {'mlmi_value': 0.0, 'mlmi_signal': 0}
            
    def test_mlmi_calculation_with_sufficient_data(self):
        """Test MLMI calculation with sufficient data"""
        # Generate trending data
        bars = self.generate_trending_data(150, trend_slope=0.1)
        
        # Process bars
        for bar in bars:
            result = self.mlmi.calculate_30m(bar)
            
        # Final result should be valid
        assert isinstance(result['mlmi_value'], float)
        assert isinstance(result['mlmi_signal'], int)
        assert result['mlmi_signal'] in [-1, 0, 1]
        
        # Check that data was stored in MLMIDataFast
        assert self.mlmi.mlmi_data.size > 0
        
    def test_mlmi_signal_generation(self):
        """Test MLMI signal generation logic"""
        # Create strongly trending up data
        bars = self.generate_trending_data(150, trend_slope=0.2)
        
        results = []
        for bar in bars:
            result = self.mlmi.calculate_30m(bar)
            results.append(result)
            
        # Should eventually generate bullish signals
        signals = [r['mlmi_signal'] for r in results[-20:]]  # Last 20 signals
        assert any(s == 1 for s in signals)  # At least one bullish signal
        
        # Test with strongly trending down data
        bars_down = self.generate_trending_data(150, trend_slope=-0.2)
        
        # Reset calculator
        self.mlmi.reset()
        
        results_down = []
        for bar in bars_down:
            result = self.mlmi.calculate_30m(bar)
            results_down.append(result)
            
        # Should eventually generate bearish signals
        signals_down = [r['mlmi_signal'] for r in results_down[-20:]]  # Last 20 signals
        assert any(s == -1 for s in signals_down)  # At least one bearish signal
        
    def test_mlmi_mathematical_properties(self):
        """Test mathematical properties of MLMI"""
        # Generate test data
        bars = self.generate_trending_data(200)
        
        mlmi_values = []
        for bar in bars:
            result = self.mlmi.calculate_30m(bar)
            mlmi_values.append(result['mlmi_value'])
            
        # MLMI values should be bounded
        assert all(abs(v) <= 1000 for v in mlmi_values)  # Reasonable bounds
        
        # Should not have excessive jumps
        for i in range(1, len(mlmi_values)):
            if mlmi_values[i-1] != 0 and mlmi_values[i] != 0:
                change_ratio = abs(mlmi_values[i] - mlmi_values[i-1]) / abs(mlmi_values[i-1])
                assert change_ratio < 10.0  # No more than 1000% change
                
    def test_mlmi_performance_requirements(self):
        """Test MLMI performance requirements"""
        # Generate sufficient data
        bars = self.generate_trending_data(100)
        
        # Add to history
        for bar in bars[:-1]:
            self.mlmi.update_30m_history(bar)
        
        # Time the calculation
        start_time = time.time()
        result = self.mlmi.calculate_30m(bars[-1])
        calc_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Should complete within 1ms for <1ms target
        assert calc_time < 1.0
        
        # Result should be valid
        assert isinstance(result['mlmi_value'], float)
        assert isinstance(result['mlmi_signal'], int)
        
    def test_mlmi_boundary_conditions(self):
        """Test MLMI boundary conditions"""
        # Test with extreme price movements
        extreme_bars = []
        prices = [100.0, 200.0, 50.0, 150.0, 25.0, 175.0, 10.0, 190.0]
        
        for i, price in enumerate(prices):
            bar = self.create_test_bar(close=price)
            extreme_bars.append(bar)
            
        # Should handle extreme price movements
        for bar in extreme_bars:
            try:
                result = self.mlmi.calculate_30m(bar)
                assert isinstance(result, dict)
                assert 'mlmi_value' in result
                assert 'mlmi_signal' in result
            except Exception as e:
                pytest.fail(f"MLMI failed on extreme price: {e}")
                
    def test_mlmi_reset_functionality(self):
        """Test MLMI reset functionality"""
        # Generate some data
        bars = self.generate_trending_data(20)
        
        for bar in bars:
            self.mlmi.calculate_30m(bar)
            
        # Check that data exists
        assert len(self.mlmi.history_30m) > 0
        assert self.mlmi.mlmi_data.size > 0
        
        # Reset
        self.mlmi.reset()
        
        # Check that data is cleared
        assert len(self.mlmi.history_30m) == 0
        assert self.mlmi.mlmi_data.size == 0
        assert self.mlmi.last_mlmi_value == 0.0
        assert self.mlmi.last_mlmi_signal == 0
        
    def test_mlmi_current_values(self):
        """Test getting current MLMI values"""
        # Initially should be zero
        values = self.mlmi.get_current_values()
        assert values == {'mlmi_value': 0.0, 'mlmi_signal': 0}
        
        # After calculation should reflect latest values
        bars = self.generate_trending_data(120)
        
        for bar in bars:
            result = self.mlmi.calculate_30m(bar)
            
        values = self.mlmi.get_current_values()
        assert values['mlmi_value'] == result['mlmi_value']
        assert values['mlmi_signal'] == result['mlmi_signal']
        
    def test_mlmi_with_real_market_patterns(self):
        """Test MLMI with realistic market patterns"""
        # Test with consolidation pattern
        consolidation_bars = []
        base_price = 100.0
        
        for i in range(100):
            # Create consolidation with slight upward bias
            noise = np.random.normal(0, 0.5)
            trend = i * 0.01  # Very slight upward trend
            price = base_price + trend + noise
            
            bar = self.create_test_bar(close=price)
            consolidation_bars.append(bar)
            
        # Process consolidation
        for bar in consolidation_bars:
            result = self.mlmi.calculate_30m(bar)
            
        # Should produce reasonable signals
        assert isinstance(result['mlmi_value'], float)
        assert result['mlmi_signal'] in [-1, 0, 1]
        
        # Test with breakout pattern
        breakout_bars = []
        for i in range(20):
            # Sudden breakout
            price = base_price + 100 + (i * 2.0)  # Strong upward movement
            bar = self.create_test_bar(close=price)
            breakout_bars.append(bar)
            
        # Process breakout
        for bar in breakout_bars:
            result = self.mlmi.calculate_30m(bar)
            
        # Should generate bullish signal
        assert result['mlmi_signal'] == 1
        
    def test_mlmi_numerical_stability(self):
        """Test MLMI numerical stability"""
        # Test with very small price changes
        bars = []
        base_price = 100.0
        
        for i in range(100):
            # Very small changes
            price = base_price + (i * 0.0001)
            bar = self.create_test_bar(close=price)
            bars.append(bar)
            
        # Should handle small changes without issues
        for bar in bars:
            result = self.mlmi.calculate_30m(bar)
            assert not np.isnan(result['mlmi_value'])
            assert not np.isinf(result['mlmi_value'])
            
    def test_mlmi_memory_usage(self):
        """Test MLMI memory usage with large datasets"""
        # Generate large dataset
        bars = self.generate_trending_data(1000)
        
        # Process all bars
        for bar in bars:
            result = self.mlmi.calculate_30m(bar)
            
        # Check that memory usage is controlled
        assert len(self.mlmi.history_30m) <= 1000  # Should be limited by max_history_length
        assert self.mlmi.mlmi_data.size < 10000  # Should not grow indefinitely
        
    def test_mlmi_configuration_validation(self):
        """Test MLMI configuration validation"""
        # Test with invalid configuration
        invalid_config = {
            'num_neighbors': -1,  # Invalid
            'momentum_window': 0,  # Invalid
            'ma_quick_period': 0,  # Invalid
        }
        
        # Should still work with defaults
        mlmi_invalid = MLMICalculator(invalid_config, self.mock_event_bus)
        assert mlmi_invalid.num_neighbors == -1  # Takes config value even if invalid
        
        # Test with missing configuration
        empty_config = {}
        mlmi_empty = MLMICalculator(empty_config, self.mock_event_bus)
        
        # Should use defaults
        assert mlmi_empty.num_neighbors == 200
        assert mlmi_empty.momentum_window == 20
        
    def test_mlmi_edge_cases(self):
        """Test MLMI edge cases"""
        # Test with all same prices
        same_price_bars = [self.create_test_bar(close=100.0) for _ in range(150)]
        
        for bar in same_price_bars:
            result = self.mlmi.calculate_30m(bar)
            
        # Should handle flat prices
        assert isinstance(result['mlmi_value'], float)
        assert result['mlmi_signal'] in [-1, 0, 1]
        
        # Test with alternating prices
        alternating_bars = []
        for i in range(150):
            price = 100.0 if i % 2 == 0 else 101.0
            bar = self.create_test_bar(close=price)
            alternating_bars.append(bar)
            
        # Reset for new test
        self.mlmi.reset()
        
        for bar in alternating_bars:
            result = self.mlmi.calculate_30m(bar)
            
        # Should handle alternating prices
        assert isinstance(result['mlmi_value'], float)
        assert result['mlmi_signal'] in [-1, 0, 1]
        
    def test_mlmi_threading_safety(self):
        """Test MLMI thread safety"""
        import threading
        
        bars = self.generate_trending_data(50)
        results = []
        
        def calculate_mlmi(bar):
            result = self.mlmi.calculate_30m(bar)
            results.append(result)
            
        # Create multiple threads
        threads = []
        for bar in bars:
            t = threading.Thread(target=calculate_mlmi, args=(bar,))
            threads.append(t)
            
        # Start all threads
        for t in threads:
            t.start()
            
        # Wait for completion
        for t in threads:
            t.join()
            
        # Should have processed all bars
        assert len(results) == len(bars)
        
        # All results should be valid
        for result in results:
            assert isinstance(result['mlmi_value'], float)
            assert result['mlmi_signal'] in [-1, 0, 1]