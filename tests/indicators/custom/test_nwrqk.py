"""
Mathematical Validation Tests for NWRQK (Nadaraya-Watson Rational Quadratic Kernel)
Tests mathematical correctness, kernel properties, and performance requirements
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import time

from src.indicators.custom.nwrqk import (
    NWRQKCalculator, 
    rational_quadratic_kernel,
    kernel_regression_numba,
    calculate_nw_regression,
    detect_crosses
)
from src.core.minimal_dependencies import EventBus, BarData
from tests.mocks.mock_event_bus import MockEventBus


class TestNWRQKCalculator:
    """Test suite for NWRQK Calculator mathematical validation"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.mock_event_bus = MockEventBus()
        self.config = {
            'h': 8.0,
            'r': 8.0,
            'x_0': 25,
            'lag': 2,
            'smooth_colors': False
        }
        self.nwrqk = NWRQKCalculator(self.config, self.mock_event_bus)
        
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
    
    def generate_synthetic_data(self, length=100, pattern_type='trend'):
        """Generate synthetic data with known patterns"""
        bars = []
        base_price = 100.0
        
        for i in range(length):
            if pattern_type == 'trend':
                # Linear trend with noise
                price = base_price + (i * 0.5) + np.random.normal(0, 0.5)
            elif pattern_type == 'sine':
                # Sine wave pattern
                price = base_price + 10 * np.sin(i * 0.1) + np.random.normal(0, 0.2)
            elif pattern_type == 'step':
                # Step function
                price = base_price + (10 if i > length//2 else 0) + np.random.normal(0, 0.1)
            elif pattern_type == 'noise':
                # Pure noise
                price = base_price + np.random.normal(0, 5.0)
            else:  # flat
                price = base_price + np.random.normal(0, 0.1)
                
            timestamp = datetime.now() + timedelta(minutes=i*30)
            bar = self.create_test_bar(close=price, timestamp=timestamp)
            bars.append(bar)
            
        return bars
    
    def test_nwrqk_initialization(self):
        """Test NWRQK calculator initialization"""
        assert self.nwrqk.h == 8.0
        assert self.nwrqk.r == 8.0
        assert self.nwrqk.x_0 == 25
        assert self.nwrqk.lag == 2
        assert self.nwrqk.smooth_colors == False
        
    def test_rational_quadratic_kernel_properties(self):
        """Test mathematical properties of the Rational Quadratic Kernel"""
        # Test kernel with same points (should be 1.0)
        kernel_same = rational_quadratic_kernel(5.0, 5.0, alpha=1.0, h=1.0)
        assert abs(kernel_same - 1.0) < 1e-10
        
        # Test kernel symmetry: K(x,y) = K(y,x)
        x, y = 3.0, 7.0
        kernel_xy = rational_quadratic_kernel(x, y, alpha=1.0, h=1.0)
        kernel_yx = rational_quadratic_kernel(y, x, alpha=1.0, h=1.0)
        assert abs(kernel_xy - kernel_yx) < 1e-10
        
        # Test kernel decreases with distance
        x = 5.0
        kernel_close = rational_quadratic_kernel(x, x + 1.0, alpha=1.0, h=1.0)
        kernel_far = rational_quadratic_kernel(x, x + 10.0, alpha=1.0, h=1.0)
        assert kernel_close > kernel_far
        
        # Test kernel is positive
        kernel_pos = rational_quadratic_kernel(0.0, 100.0, alpha=1.0, h=1.0)
        assert kernel_pos > 0.0
        
        # Test kernel bounds (should be between 0 and 1)
        for i in range(10):
            x1 = np.random.uniform(-100, 100)
            x2 = np.random.uniform(-100, 100)
            kernel_val = rational_quadratic_kernel(x1, x2, alpha=1.0, h=1.0)
            assert 0.0 <= kernel_val <= 1.0
            
    def test_rational_quadratic_kernel_parameters(self):
        """Test RQ kernel parameter effects"""
        x1, x2 = 0.0, 5.0
        
        # Test alpha parameter effect
        kernel_alpha_small = rational_quadratic_kernel(x1, x2, alpha=0.5, h=1.0)
        kernel_alpha_large = rational_quadratic_kernel(x1, x2, alpha=2.0, h=1.0)
        
        # Both should be valid
        assert 0.0 <= kernel_alpha_small <= 1.0
        assert 0.0 <= kernel_alpha_large <= 1.0
        
        # Test h parameter effect (bandwidth)
        kernel_h_small = rational_quadratic_kernel(x1, x2, alpha=1.0, h=0.5)
        kernel_h_large = rational_quadratic_kernel(x1, x2, alpha=1.0, h=2.0)
        
        # Smaller h should give smaller kernel values for distant points
        assert kernel_h_small < kernel_h_large
        
    def test_kernel_regression_numba_basic(self):
        """Test basic kernel regression functionality"""
        # Test with simple data
        src = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        size = len(src) - 1
        h_param = 1.0
        r_param = 1.0
        x_0_param = 0
        
        result = kernel_regression_numba(src, size, h_param, r_param, x_0_param)
        
        # Should return a valid number
        assert not np.isnan(result)
        assert not np.isinf(result)
        assert isinstance(result, float)
        
    def test_kernel_regression_numba_edge_cases(self):
        """Test kernel regression edge cases"""
        # Test with empty array
        empty_src = np.array([])
        result_empty = kernel_regression_numba(empty_src, 0, 1.0, 1.0, 0)
        assert np.isnan(result_empty)
        
        # Test with single point
        single_src = np.array([5.0])
        result_single = kernel_regression_numba(single_src, 0, 1.0, 1.0, 0)
        assert not np.isnan(result_single)
        
        # Test with all same values
        same_src = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
        result_same = kernel_regression_numba(same_src, 4, 1.0, 1.0, 0)
        assert abs(result_same - 3.0) < 1e-10
        
    def test_calculate_nw_regression(self):
        """Test NW regression calculation"""
        # Create test data
        prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
        h_param = 1.0
        h_lag_param = 1.0
        r_param = 1.0
        x_0_param = 2
        
        yhat1, yhat2 = calculate_nw_regression(prices, h_param, h_lag_param, r_param, x_0_param)
        
        # Should return arrays of correct length
        assert len(yhat1) == len(prices)
        assert len(yhat2) == len(prices)
        
        # Early values should be NaN (before x_0)
        assert np.isnan(yhat1[0])
        assert np.isnan(yhat1[1])
        
        # Later values should be valid
        assert not np.isnan(yhat1[-1])
        assert not np.isnan(yhat2[-1])
        
    def test_detect_crosses(self):
        """Test crossover detection"""
        # Create test data with known crossovers
        n = 10
        yhat1 = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        yhat2 = np.array([0.5, 0.7, 0.9, 1.1, 1.3, 1.1, 0.9, 0.7, 0.5, 0.3])
        
        bullish_cross, bearish_cross = detect_crosses(yhat1, yhat2)
        
        # Should detect bullish cross around index 3 (yhat2 crosses above yhat1)
        assert bullish_cross[3] == True
        
        # Should detect bearish cross around index 5 (yhat2 crosses below yhat1)
        assert bearish_cross[5] == True
        
        # First element should be False (no previous value to compare)
        assert bullish_cross[0] == False
        assert bearish_cross[0] == False
        
    def test_nwrqk_calculation_insufficient_data(self):
        """Test NWRQK calculation with insufficient data"""
        # Test with no data
        result = self.nwrqk.calculate_30m(self.create_test_bar())
        assert result == {'nwrqk_value': 0.0, 'nwrqk_signal': 0}
        
        # Test with insufficient data (< 50 bars)
        for i in range(30):
            bar = self.create_test_bar(close=100.0 + i)
            result = self.nwrqk.calculate_30m(bar)
            assert result == {'nwrqk_value': 0.0, 'nwrqk_signal': 0}
            
    def test_nwrqk_calculation_sufficient_data(self):
        """Test NWRQK calculation with sufficient data"""
        # Generate trending data
        bars = self.generate_synthetic_data(80, 'trend')
        
        # Process bars
        for bar in bars:
            result = self.nwrqk.calculate_30m(bar)
            
        # Final result should be valid
        assert isinstance(result['nwrqk_value'], float)
        assert isinstance(result['nwrqk_signal'], int)
        assert result['nwrqk_signal'] in [-1, 0, 1]
        assert not np.isnan(result['nwrqk_value'])
        assert not np.isinf(result['nwrqk_value'])
        
    def test_nwrqk_trend_detection(self):
        """Test NWRQK trend detection capabilities"""
        # Test with strong uptrend
        uptrend_bars = self.generate_synthetic_data(100, 'trend')
        
        signals = []
        for bar in uptrend_bars:
            result = self.nwrqk.calculate_30m(bar)
            signals.append(result['nwrqk_signal'])
            
        # Should generate some bullish signals for uptrend
        assert any(s == 1 for s in signals[-20:])  # At least one bullish signal in last 20
        
        # Test with sine wave (should generate alternating signals)
        self.nwrqk.reset()
        sine_bars = self.generate_synthetic_data(100, 'sine')
        
        sine_signals = []
        for bar in sine_bars:
            result = self.nwrqk.calculate_30m(bar)
            sine_signals.append(result['nwrqk_signal'])
            
        # Sine wave should generate both bullish and bearish signals
        assert any(s == 1 for s in sine_signals)
        assert any(s == -1 for s in sine_signals)
        
    def test_nwrqk_crossover_signals(self):
        """Test NWRQK crossover signal generation"""
        # Create data with clear trend change
        bars = []
        
        # First phase: declining trend
        for i in range(50):
            price = 120.0 - (i * 0.3)
            bars.append(self.create_test_bar(close=price))
            
        # Second phase: rising trend
        for i in range(50):
            price = 105.0 + (i * 0.4)
            bars.append(self.create_test_bar(close=price))
            
        signals = []
        for bar in bars:
            result = self.nwrqk.calculate_30m(bar)
            signals.append(result['nwrqk_signal'])
            
        # Should detect the trend change
        assert any(s != 0 for s in signals)
        
    def test_nwrqk_performance_requirements(self):
        """Test NWRQK performance requirements"""
        # Generate sufficient data
        bars = self.generate_synthetic_data(60)
        
        # Add to history
        for bar in bars[:-1]:
            self.nwrqk.update_30m_history(bar)
            
        # Time the calculation
        start_time = time.time()
        result = self.nwrqk.calculate_30m(bars[-1])
        calc_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Should complete within 1ms for <1ms target
        assert calc_time < 1.0
        
        # Result should be valid
        assert isinstance(result['nwrqk_value'], float)
        assert isinstance(result['nwrqk_signal'], int)
        
    def test_nwrqk_mathematical_properties(self):
        """Test mathematical properties of NWRQK"""
        # Generate test data
        bars = self.generate_synthetic_data(100, 'trend')
        
        values = []
        for bar in bars:
            result = self.nwrqk.calculate_30m(bar)
            values.append(result['nwrqk_value'])
            
        # Values should be reasonable (not extreme)
        valid_values = [v for v in values if not np.isnan(v)]
        if valid_values:
            mean_val = np.mean(valid_values)
            std_val = np.std(valid_values)
            
            # Should not have extreme outliers
            for val in valid_values:
                assert abs(val - mean_val) < 10 * std_val
                
    def test_nwrqk_boundary_conditions(self):
        """Test NWRQK boundary conditions"""
        # Test with extreme price movements
        extreme_bars = []
        prices = [100.0, 1000.0, 10.0, 500.0, 1.0, 750.0]
        
        for price in prices:
            bar = self.create_test_bar(close=price)
            extreme_bars.append(bar)
            
        # Should handle extreme movements
        for bar in extreme_bars:
            try:
                result = self.nwrqk.calculate_30m(bar)
                assert isinstance(result, dict)
                assert 'nwrqk_value' in result
                assert 'nwrqk_signal' in result
            except Exception as e:
                pytest.fail(f"NWRQK failed on extreme price: {e}")
                
    def test_nwrqk_parameter_sensitivity(self):
        """Test NWRQK sensitivity to parameters"""
        # Test different h values
        configs = [
            {'h': 4.0, 'r': 8.0, 'x_0': 25, 'lag': 2},
            {'h': 8.0, 'r': 8.0, 'x_0': 25, 'lag': 2},
            {'h': 16.0, 'r': 8.0, 'x_0': 25, 'lag': 2}
        ]
        
        bars = self.generate_synthetic_data(80, 'trend')
        results = []
        
        for config in configs:
            nwrqk = NWRQKCalculator(config, self.mock_event_bus)
            
            for bar in bars:
                result = nwrqk.calculate_30m(bar)
                
            results.append(result)
            
        # Different parameters should give different results
        assert results[0]['nwrqk_value'] != results[1]['nwrqk_value']
        assert results[1]['nwrqk_value'] != results[2]['nwrqk_value']
        
    def test_nwrqk_reset_functionality(self):
        """Test NWRQK reset functionality"""
        # Generate some data
        bars = self.generate_synthetic_data(30)
        
        for bar in bars:
            self.nwrqk.calculate_30m(bar)
            
        # Check that data exists
        assert len(self.nwrqk.history_30m) > 0
        
        # Reset
        self.nwrqk.reset()
        
        # Check that data is cleared
        assert len(self.nwrqk.history_30m) == 0
        
    def test_nwrqk_smooth_colors_mode(self):
        """Test NWRQK smooth colors mode"""
        # Test with smooth_colors enabled
        smooth_config = self.config.copy()
        smooth_config['smooth_colors'] = True
        
        nwrqk_smooth = NWRQKCalculator(smooth_config, self.mock_event_bus)
        
        # Generate data
        bars = self.generate_synthetic_data(80, 'sine')
        
        # Process bars
        for bar in bars:
            result = nwrqk_smooth.calculate_30m(bar)
            
        # Should still produce valid results
        assert isinstance(result['nwrqk_value'], float)
        assert isinstance(result['nwrqk_signal'], int)
        assert result['nwrqk_signal'] in [-1, 0, 1]
        
    def test_nwrqk_numerical_stability(self):
        """Test NWRQK numerical stability"""
        # Test with very small price changes
        bars = []
        base_price = 100.0
        
        for i in range(60):
            # Very small changes
            price = base_price + (i * 0.00001)
            bar = self.create_test_bar(close=price)
            bars.append(bar)
            
        # Should handle small changes without issues
        for bar in bars:
            result = self.nwrqk.calculate_30m(bar)
            assert not np.isnan(result['nwrqk_value'])
            assert not np.isinf(result['nwrqk_value'])
            
    def test_nwrqk_lag_parameter_effect(self):
        """Test effect of lag parameter"""
        # Test different lag values
        configs = [
            {'h': 8.0, 'r': 8.0, 'x_0': 25, 'lag': 1},
            {'h': 8.0, 'r': 8.0, 'x_0': 25, 'lag': 2},
            {'h': 8.0, 'r': 8.0, 'x_0': 25, 'lag': 3}
        ]
        
        bars = self.generate_synthetic_data(80, 'trend')
        results = []
        
        for config in configs:
            nwrqk = NWRQKCalculator(config, self.mock_event_bus)
            
            for bar in bars:
                result = nwrqk.calculate_30m(bar)
                
            results.append(result)
            
        # Different lag values should potentially give different results
        # (though they might be similar for smooth trends)
        assert all(isinstance(r['nwrqk_value'], float) for r in results)
        assert all(r['nwrqk_signal'] in [-1, 0, 1] for r in results)
        
    def test_nwrqk_memory_efficiency(self):
        """Test NWRQK memory efficiency"""
        # Generate large dataset
        bars = self.generate_synthetic_data(500)
        
        # Process all bars
        for bar in bars:
            result = self.nwrqk.calculate_30m(bar)
            
        # Check that memory usage is controlled
        assert len(self.nwrqk.history_30m) <= 1000  # Should be limited by max_history_length
        
    def test_nwrqk_signal_consistency(self):
        """Test NWRQK signal consistency"""
        # Generate consistent trend data
        bars = self.generate_synthetic_data(100, 'trend')
        
        signals = []
        for bar in bars:
            result = self.nwrqk.calculate_30m(bar)
            signals.append(result['nwrqk_signal'])
            
        # Check that signals are not too noisy
        signal_changes = sum(1 for i in range(1, len(signals)) if signals[i] != signals[i-1])
        
        # Should not change signals too frequently
        assert signal_changes < len(signals) * 0.5  # Less than 50% changes
        
    def test_nwrqk_edge_case_flat_prices(self):
        """Test NWRQK with flat prices"""
        # Test with all same prices
        flat_bars = [self.create_test_bar(close=100.0) for _ in range(80)]
        
        for bar in flat_bars:
            result = self.nwrqk.calculate_30m(bar)
            
        # Should handle flat prices gracefully
        assert isinstance(result['nwrqk_value'], float)
        assert result['nwrqk_signal'] in [-1, 0, 1]
        assert not np.isnan(result['nwrqk_value'])
        
    def test_nwrqk_current_values(self):
        """Test getting current NWRQK values"""
        # Initially should return defaults
        values = self.nwrqk.get_current_values()
        assert values == {'nwrqk_value': 0.0, 'nwrqk_signal': 0}
        
    def test_nwrqk_dataframe_processing(self):
        """Test NWRQK DataFrame processing logic"""
        # Generate test data
        bars = self.generate_synthetic_data(60, 'trend')
        
        # Process bars to build history
        for bar in bars:
            result = self.nwrqk.calculate_30m(bar)
            
        # Check that DataFrame processing worked
        assert len(self.nwrqk.history_30m) > 0
        
        # Final result should be valid
        assert isinstance(result['nwrqk_value'], float)
        assert isinstance(result['nwrqk_signal'], int)
        
    def test_nwrqk_threading_safety(self):
        """Test NWRQK thread safety"""
        import threading
        
        bars = self.generate_synthetic_data(30)
        results = []
        
        def calculate_nwrqk(bar):
            result = self.nwrqk.calculate_30m(bar)
            results.append(result)
            
        # Create multiple threads
        threads = []
        for bar in bars:
            t = threading.Thread(target=calculate_nwrqk, args=(bar,))
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
            assert isinstance(result['nwrqk_value'], float)
            assert result['nwrqk_signal'] in [-1, 0, 1]