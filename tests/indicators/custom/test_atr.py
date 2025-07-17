"""
Mathematical Validation Tests for ATR (Average True Range) Indicator
Tests True Range calculations, ATR computation, and volatility metrics
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import time

from src.indicators.custom.atr import ATRIndicator, ATRReading
from src.core.minimal_dependencies import EventBus, BarData
from tests.mocks.mock_event_bus import MockEventBus


class TestATRIndicator:
    """Test suite for ATR Indicator mathematical validation"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.mock_event_bus = MockEventBus()
        self.config = {
            'period': 14,
            'smoothing_factor': 0.1,
            'min_periods': 2,
            'vol_low_percentile': 25,
            'vol_medium_percentile': 50,
            'vol_high_percentile': 75
        }
        self.atr = ATRIndicator(self.config, self.mock_event_bus)
        
    def create_test_bar(self, open_price, high, low, close, volume=1000, timestamp=None):
        """Create a test bar with specific OHLC values"""
        if timestamp is None:
            timestamp = datetime.now()
            
        return BarData(
            symbol="BTCUSDT",
            timestamp=timestamp,
            open=open_price,
            high=high,
            low=low,
            close=close,
            volume=volume,
            timeframe=30
        )
    
    def create_volatility_series(self, length=50, volatility_type='normal'):
        """Create bars with specific volatility characteristics"""
        bars = []
        base_price = 100.0
        
        for i in range(length):
            if volatility_type == 'low':
                # Low volatility: small ranges
                price = base_price + np.random.normal(0, 0.1)
                high = price + np.random.uniform(0.05, 0.2)
                low = price - np.random.uniform(0.05, 0.2)
            elif volatility_type == 'high':
                # High volatility: large ranges
                price = base_price + np.random.normal(0, 2.0)
                high = price + np.random.uniform(1.0, 5.0)
                low = price - np.random.uniform(1.0, 5.0)
            else:  # normal
                # Normal volatility
                price = base_price + np.random.normal(0, 0.5)
                high = price + np.random.uniform(0.2, 1.0)
                low = price - np.random.uniform(0.2, 1.0)
                
            # Ensure high >= low
            if high < low:
                high, low = low, high
                
            open_price = price + np.random.uniform(-0.5, 0.5)
            
            timestamp = datetime.now() + timedelta(minutes=i*30)
            bar = self.create_test_bar(open_price, high, low, price, timestamp=timestamp)
            bars.append(bar)
            
        return bars
    
    def test_atr_initialization(self):
        """Test ATR indicator initialization"""
        assert self.atr.period == 14
        assert self.atr.smoothing_factor == 0.1
        assert self.atr.min_periods == 2
        assert self.atr.current_atr is None
        assert self.atr.current_volatility_percentile is None
        assert self.atr.ewma_atr is None
        assert len(self.atr.price_data) == 0
        assert len(self.atr.true_ranges) == 0
        assert len(self.atr.atr_values) == 0
        assert self.atr.calculations_performed == 0
        
    def test_true_range_calculation(self):
        """Test True Range calculation"""
        # Test case 1: High-Low is largest
        current_bar = self.create_test_bar(100.0, 105.0, 95.0, 102.0)
        previous_close = 100.0
        
        tr = self.atr.calculate_true_range(current_bar, previous_close)
        assert tr == 10.0  # 105.0 - 95.0 = 10.0
        
        # Test case 2: High-PrevClose is largest
        current_bar = self.create_test_bar(100.0, 108.0, 99.0, 102.0)
        previous_close = 95.0
        
        tr = self.atr.calculate_true_range(current_bar, previous_close)
        assert tr == 13.0  # 108.0 - 95.0 = 13.0
        
        # Test case 3: PrevClose-Low is largest
        current_bar = self.create_test_bar(100.0, 102.0, 92.0, 101.0)
        previous_close = 105.0
        
        tr = self.atr.calculate_true_range(current_bar, previous_close)
        assert tr == 13.0  # 105.0 - 92.0 = 13.0
        
        # Test case 4: All equal (edge case)
        current_bar = self.create_test_bar(100.0, 100.0, 100.0, 100.0)
        previous_close = 100.0
        
        tr = self.atr.calculate_true_range(current_bar, previous_close)
        assert tr == 0.0
        
    def test_atr_calculation_basic(self):
        """Test basic ATR calculation"""
        # Test with insufficient data
        true_ranges = [5.0]
        atr = self.atr.calculate_atr(true_ranges)
        assert atr == 0.0  # Insufficient data
        
        # Test with minimum data
        true_ranges = [5.0, 3.0]
        atr = self.atr.calculate_atr(true_ranges)
        assert atr == 4.0  # (5.0 + 3.0) / 2
        
        # Test with full period
        true_ranges = [i + 1.0 for i in range(14)]  # 1.0 to 14.0
        atr = self.atr.calculate_atr(true_ranges)
        expected = sum(true_ranges) / 14
        assert abs(atr - expected) < 1e-10
        
        # Test with more than period
        true_ranges = [i + 1.0 for i in range(20)]  # 1.0 to 20.0
        atr = self.atr.calculate_atr(true_ranges)
        expected = sum(true_ranges[-14:]) / 14  # Last 14 values
        assert abs(atr - expected) < 1e-10
        
    def test_ewma_atr_calculation(self):
        """Test EWMA ATR calculation"""
        # First value should initialize EWMA
        ewma1 = self.atr.calculate_ewma_atr(10.0)
        assert ewma1 == 10.0
        
        # Second value should use EWMA formula
        ewma2 = self.atr.calculate_ewma_atr(20.0)
        expected = 0.1 * 20.0 + 0.9 * 10.0
        assert abs(ewma2 - expected) < 1e-10
        
        # Third value should continue EWMA
        ewma3 = self.atr.calculate_ewma_atr(5.0)
        expected = 0.1 * 5.0 + 0.9 * ewma2
        assert abs(ewma3 - expected) < 1e-10
        
    def test_volatility_percentile_calculation(self):
        """Test volatility percentile calculation"""
        # Test with insufficient data
        percentile = self.atr.calculate_volatility_percentile(5.0)
        assert percentile == 50.0  # Default neutral
        
        # Test with sufficient data
        atr_values = [1.0, 2.0, 3.0, 4.0, 5.0] * 5  # 25 values
        self.atr.atr_values = atr_values
        
        percentile = self.atr.calculate_volatility_percentile(3.0)
        assert 0.0 <= percentile <= 100.0
        
        # Test with extreme values
        percentile_low = self.atr.calculate_volatility_percentile(0.5)
        percentile_high = self.atr.calculate_volatility_percentile(6.0)
        assert percentile_low < percentile_high
        
        # Test with zero or negative current ATR
        percentile_zero = self.atr.calculate_volatility_percentile(0.0)
        assert percentile_zero == 0.0
        
        percentile_negative = self.atr.calculate_volatility_percentile(-1.0)
        assert percentile_negative == 0.0
        
    def test_trend_strength_calculation(self):
        """Test trend strength calculation"""
        # Test with insufficient data
        strength = self.atr.calculate_trend_strength([])
        assert strength == 0.0
        
        # Test with insufficient bars
        bars = [self.create_test_bar(100.0, 102.0, 98.0, 101.0)]
        strength = self.atr.calculate_trend_strength(bars)
        assert strength == 0.0
        
        # Test with no ATR
        bars = [
            self.create_test_bar(100.0, 102.0, 98.0, 101.0),
            self.create_test_bar(101.0, 103.0, 99.0, 102.0),
            self.create_test_bar(102.0, 104.0, 100.0, 103.0)
        ]
        strength = self.atr.calculate_trend_strength(bars)
        assert strength == 0.0
        
        # Test with valid data
        self.atr.current_atr = 2.0
        strength = self.atr.calculate_trend_strength(bars)
        assert 0.0 <= strength <= 1.0
        
        # Test with strong trend
        strong_trend_bars = [
            self.create_test_bar(100.0, 102.0, 98.0, 101.0),
            self.create_test_bar(101.0, 103.0, 99.0, 102.0),
            self.create_test_bar(102.0, 114.0, 100.0, 113.0)  # Big move
        ]
        strong_strength = self.atr.calculate_trend_strength(strong_trend_bars)
        assert strong_strength > strength
        
    def test_atr_update_process(self):
        """Test complete ATR update process"""
        # Create test bars
        bars = self.create_volatility_series(20, 'normal')
        
        # Process first bar (should return None - insufficient data)
        reading1 = self.atr.update(bars[0])
        assert reading1 is None
        
        # Process second bar (should return valid reading)
        reading2 = self.atr.update(bars[1])
        assert reading2 is not None
        assert isinstance(reading2, ATRReading)
        assert reading2.atr_value > 0.0
        assert reading2.true_range >= 0.0
        assert 0.0 <= reading2.volatility_percentile <= 100.0
        assert 0.0 <= reading2.trend_strength <= 1.0
        
        # Process more bars
        for bar in bars[2:]:
            reading = self.atr.update(bar)
            assert reading is not None
            assert isinstance(reading, ATRReading)
            
        # Check final state
        assert len(self.atr.price_data) == 20
        assert len(self.atr.true_ranges) == 19  # n-1 true ranges
        assert len(self.atr.atr_values) == 19
        assert self.atr.calculations_performed == 19
        assert self.atr.current_atr is not None
        assert self.atr.current_atr > 0.0
        
    def test_atr_volatility_regimes(self):
        """Test ATR volatility regime classification"""
        # Test with low volatility data
        low_vol_bars = self.create_volatility_series(30, 'low')
        for bar in low_vol_bars:
            self.atr.update(bar)
            
        regime_low = self.atr.get_volatility_regime()
        
        # Test with high volatility data
        self.atr.reset()
        high_vol_bars = self.create_volatility_series(30, 'high')
        for bar in high_vol_bars:
            self.atr.update(bar)
            
        regime_high = self.atr.get_volatility_regime()
        
        # High volatility should give higher regime
        regimes = ['low', 'medium', 'high', 'extreme']
        assert regimes.index(regime_high) >= regimes.index(regime_low)
        
    def test_atr_recommendations(self):
        """Test ATR-based recommendations"""
        # Process some data
        bars = self.create_volatility_series(20, 'normal')
        for bar in bars:
            self.atr.update(bar)
            
        # Test stop distance recommendation
        stop_distance = self.atr.get_stop_distance_recommendation(1.5)
        assert stop_distance is not None
        assert stop_distance > 0.0
        assert stop_distance == self.atr.current_atr * 1.5
        
        # Test target distance recommendation
        target_distance = self.atr.get_target_distance_recommendation(2.0)
        assert target_distance is not None
        assert target_distance > 0.0
        assert target_distance == self.atr.current_atr * 2.0
        
        # Test with no ATR
        self.atr.current_atr = None
        assert self.atr.get_stop_distance_recommendation() is None
        assert self.atr.get_target_distance_recommendation() is None
        
    def test_atr_statistics(self):
        """Test ATR statistics calculation"""
        # Test with no data
        stats = self.atr.get_statistics()
        assert stats == {}
        
        # Test with data
        bars = self.create_volatility_series(25, 'normal')
        for bar in bars:
            self.atr.update(bar)
            
        stats = self.atr.get_statistics()
        
        # Check required fields
        assert 'current_atr' in stats
        assert 'ewma_atr' in stats
        assert 'volatility_percentile' in stats
        assert 'volatility_regime' in stats
        assert 'calculations_performed' in stats
        assert 'last_update' in stats
        assert 'atr_statistics' in stats
        
        # Check ATR statistics
        atr_stats = stats['atr_statistics']
        assert 'mean' in atr_stats
        assert 'std' in atr_stats
        assert 'min' in atr_stats
        assert 'max' in atr_stats
        assert 'latest' in atr_stats
        
        # Check values are reasonable
        assert stats['current_atr'] > 0.0
        assert stats['ewma_atr'] > 0.0
        assert 0.0 <= stats['volatility_percentile'] <= 100.0
        assert stats['volatility_regime'] in ['low', 'medium', 'high', 'extreme']
        assert stats['calculations_performed'] > 0
        
    def test_atr_reset(self):
        """Test ATR reset functionality"""
        # Process some data
        bars = self.create_volatility_series(10, 'normal')
        for bar in bars:
            self.atr.update(bar)
            
        # Verify data exists
        assert len(self.atr.price_data) > 0
        assert len(self.atr.true_ranges) > 0
        assert len(self.atr.atr_values) > 0
        assert self.atr.current_atr is not None
        
        # Reset
        self.atr.reset()
        
        # Verify reset
        assert len(self.atr.price_data) == 0
        assert len(self.atr.true_ranges) == 0
        assert len(self.atr.atr_values) == 0
        assert len(self.atr.atr_readings) == 0
        assert self.atr.current_atr is None
        assert self.atr.current_volatility_percentile is None
        assert self.atr.ewma_atr is None
        assert self.atr.calculations_performed == 0
        assert self.atr.last_update_time is None
        
    def test_atr_input_validation(self):
        """Test ATR input validation"""
        # Test None input
        assert self.atr.validate_inputs(None) == False
        
        # Test valid input
        valid_bar = self.create_test_bar(100.0, 102.0, 98.0, 101.0)
        assert self.atr.validate_inputs(valid_bar) == True
        
        # Test NaN values
        nan_bar = self.create_test_bar(float('nan'), 102.0, 98.0, 101.0)
        assert self.atr.validate_inputs(nan_bar) == False
        
        # Test infinity values
        inf_bar = self.create_test_bar(100.0, float('inf'), 98.0, 101.0)
        assert self.atr.validate_inputs(inf_bar) == False
        
        # Test high < low
        invalid_bar = self.create_test_bar(100.0, 95.0, 98.0, 101.0)
        assert self.atr.validate_inputs(invalid_bar) == False
        
        # Test negative close
        negative_bar = self.create_test_bar(100.0, 102.0, 98.0, -101.0)
        assert self.atr.validate_inputs(negative_bar) == False
        
    def test_atr_performance_requirements(self):
        """Test ATR performance requirements"""
        # Generate sufficient data
        bars = self.create_volatility_series(20, 'normal')
        
        # Add to history
        for bar in bars[:-1]:
            self.atr.update(bar)
            
        # Time the calculation
        start_time = time.time()
        reading = self.atr.update(bars[-1])
        calc_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Should complete within 5ms
        assert calc_time < 5.0
        
        # Result should be valid
        assert reading is not None
        assert isinstance(reading, ATRReading)
        assert reading.atr_value > 0.0
        
    def test_atr_mathematical_properties(self):
        """Test mathematical properties of ATR"""
        # Generate test data
        bars = self.create_volatility_series(50, 'normal')
        
        readings = []
        for bar in bars:
            reading = self.atr.update(bar)
            if reading is not None:
                readings.append(reading)
                
        # ATR should be non-negative
        assert all(r.atr_value >= 0.0 for r in readings)
        
        # True Range should be non-negative
        assert all(r.true_range >= 0.0 for r in readings)
        
        # Volatility percentile should be 0-100
        assert all(0.0 <= r.volatility_percentile <= 100.0 for r in readings)
        
        # Trend strength should be 0-1
        assert all(0.0 <= r.trend_strength <= 1.0 for r in readings)
        
        # ATR should be smoothed (not too volatile)
        atr_values = [r.atr_value for r in readings]
        if len(atr_values) > 1:
            # ATR shouldn't change too dramatically
            max_change = max(abs(atr_values[i] - atr_values[i-1]) 
                           for i in range(1, len(atr_values)))
            avg_atr = np.mean(atr_values)
            assert max_change < avg_atr * 2.0  # Max change < 200% of average
            
    def test_atr_edge_cases(self):
        """Test ATR edge cases"""
        # Test with zero ranges
        zero_bars = []
        for i in range(10):
            bar = self.create_test_bar(100.0, 100.0, 100.0, 100.0)
            zero_bars.append(bar)
            
        for bar in zero_bars:
            reading = self.atr.update(bar)
            if reading is not None:
                assert reading.true_range == 0.0
                assert reading.atr_value >= 0.0
                
        # Test with extreme ranges
        extreme_bars = []
        for i in range(10):
            bar = self.create_test_bar(100.0, 200.0, 50.0, 150.0)
            extreme_bars.append(bar)
            
        self.atr.reset()
        for bar in extreme_bars:
            reading = self.atr.update(bar)
            if reading is not None:
                assert reading.true_range > 0.0
                assert reading.atr_value > 0.0
                
    def test_atr_memory_management(self):
        """Test ATR memory management"""
        # Generate large dataset
        bars = self.create_volatility_series(500)
        
        # Process all bars
        for bar in bars:
            self.atr.update(bar)
            
        # Check memory limits
        max_history = max(self.atr.period * 3, 100)
        assert len(self.atr.price_data) <= max_history
        assert len(self.atr.true_ranges) <= max_history
        assert len(self.atr.atr_values) <= max_history
        assert len(self.atr.atr_readings) <= 1000
        
    def test_atr_configuration_validation(self):
        """Test ATR configuration validation"""
        # Test with custom configuration
        custom_config = {
            'period': 20,
            'smoothing_factor': 0.05,
            'min_periods': 3,
            'vol_low_percentile': 20,
            'vol_medium_percentile': 60,
            'vol_high_percentile': 80
        }
        
        atr_custom = ATRIndicator(custom_config, self.mock_event_bus)
        
        assert atr_custom.period == 20
        assert atr_custom.smoothing_factor == 0.05
        assert atr_custom.min_periods == 3
        assert atr_custom.volatility_thresholds['low'] == 20
        assert atr_custom.volatility_thresholds['medium'] == 60
        assert atr_custom.volatility_thresholds['high'] == 80
        
    def test_atr_string_representations(self):
        """Test ATR string representations"""
        # Test string representation
        str_repr = str(self.atr)
        assert 'ATR' in str_repr
        assert 'period=14' in str_repr
        
        # Test detailed representation
        detailed_repr = repr(self.atr)
        assert 'ATRIndicator' in detailed_repr
        assert 'period=14' in detailed_repr
        assert 'calculations=0' in detailed_repr
        
    def test_atr_threading_safety(self):
        """Test ATR thread safety"""
        import threading
        
        bars = self.create_volatility_series(20)
        readings = []
        
        def update_atr(bar):
            reading = self.atr.update(bar)
            if reading is not None:
                readings.append(reading)
                
        # Create multiple threads
        threads = []
        for bar in bars:
            t = threading.Thread(target=update_atr, args=(bar,))
            threads.append(t)
            
        # Start all threads
        for t in threads:
            t.start()
            
        # Wait for completion
        for t in threads:
            t.join()
            
        # Should have processed bars (some may be None due to insufficient data)
        assert len(readings) > 0
        
        # All readings should be valid
        for reading in readings:
            assert isinstance(reading, ATRReading)
            assert reading.atr_value >= 0.0
            assert reading.true_range >= 0.0
            
    def test_atr_stress_testing(self):
        """Test ATR under stress conditions"""
        # Generate large dataset with varying volatility
        bars = []
        for i in range(200):
            if i % 50 < 25:
                vol_type = 'low'
            else:
                vol_type = 'high'
                
            bar_set = self.create_volatility_series(1, vol_type)
            bars.extend(bar_set)
            
        # Process all bars - should not crash
        for bar in bars:
            reading = self.atr.update(bar)
            if reading is not None:
                assert isinstance(reading, ATRReading)
                assert reading.atr_value >= 0.0
                assert reading.true_range >= 0.0
                assert 0.0 <= reading.volatility_percentile <= 100.0
                assert 0.0 <= reading.trend_strength <= 1.0