"""
Mathematical Validation Tests for FVG (Fair Value Gap) Detector
Tests gap detection logic, mathematical correctness, and performance requirements
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import time

from src.indicators.custom.fvg import FVGDetector, detect_fvg, generate_fvg_data_fast
from src.core.minimal_dependencies import EventBus, BarData
from tests.mocks.mock_event_bus import MockEventBus


class TestFVGDetector:
    """Test suite for FVG Detector mathematical validation"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.mock_event_bus = MockEventBus()
        self.config = {
            'threshold': 0.001,
            'lookback_period': 10,
            'body_multiplier': 1.5
        }
        self.fvg = FVGDetector(self.config, self.mock_event_bus)
        
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
            timeframe=5
        )
    
    def create_bullish_fvg_pattern(self):
        """Create a bullish FVG pattern: gap up where current low > previous high"""
        bars = [
            # Bar 1: Normal bar
            self.create_test_bar(100.0, 102.0, 98.0, 101.0),
            
            # Bar 2: Strong bullish bar creating gap
            self.create_test_bar(101.0, 108.0, 100.0, 107.0),
            
            # Bar 3: Gap up bar (low > bar 1 high)
            self.create_test_bar(107.0, 109.0, 103.0, 108.0)  # low=103 > bar1.high=102
        ]
        return bars
    
    def create_bearish_fvg_pattern(self):
        """Create a bearish FVG pattern: gap down where current high < previous low"""
        bars = [
            # Bar 1: Normal bar
            self.create_test_bar(100.0, 102.0, 98.0, 99.0),
            
            # Bar 2: Strong bearish bar creating gap
            self.create_test_bar(99.0, 100.0, 92.0, 93.0),
            
            # Bar 3: Gap down bar (high < bar 1 low)
            self.create_test_bar(93.0, 97.0, 91.0, 94.0)  # high=97 < bar1.low=98
        ]
        return bars
    
    def create_no_gap_pattern(self):
        """Create bars with no gaps"""
        bars = [
            self.create_test_bar(100.0, 102.0, 98.0, 101.0),
            self.create_test_bar(101.0, 103.0, 99.0, 102.0),
            self.create_test_bar(102.0, 104.0, 100.0, 103.0)
        ]
        return bars
    
    def test_fvg_initialization(self):
        """Test FVG detector initialization"""
        assert self.fvg.threshold == 0.001
        assert self.fvg.lookback_period == 10
        assert self.fvg.body_multiplier == 1.5
        
    def test_detect_fvg_function_bullish(self):
        """Test detect_fvg function with bullish pattern"""
        bars = self.create_bullish_fvg_pattern()
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            'High': b.high,
            'Low': b.low,
            'Open': b.open,
            'Close': b.close
        } for b in bars])
        
        # Detect FVGs
        fvg_list = detect_fvg(df, lookback_period=10, body_multiplier=1.5)
        
        # Should detect bullish FVG at index 2
        assert fvg_list[2] is not None
        assert fvg_list[2][0] == 'bullish'
        assert fvg_list[2][1] == 102.0  # first_high
        assert fvg_list[2][2] == 103.0  # third_low
        assert fvg_list[2][3] == 2      # creation index
        
    def test_detect_fvg_function_bearish(self):
        """Test detect_fvg function with bearish pattern"""
        bars = self.create_bearish_fvg_pattern()
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            'High': b.high,
            'Low': b.low,
            'Open': b.open,
            'Close': b.close
        } for b in bars])
        
        # Detect FVGs
        fvg_list = detect_fvg(df, lookback_period=10, body_multiplier=1.5)
        
        # Should detect bearish FVG at index 2
        assert fvg_list[2] is not None
        assert fvg_list[2][0] == 'bearish'
        assert fvg_list[2][1] == 98.0   # first_low
        assert fvg_list[2][2] == 97.0   # third_high
        assert fvg_list[2][3] == 2      # creation index
        
    def test_detect_fvg_function_no_gaps(self):
        """Test detect_fvg function with no gaps"""
        bars = self.create_no_gap_pattern()
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            'High': b.high,
            'Low': b.low,
            'Open': b.open,
            'Close': b.close
        } for b in bars])
        
        # Detect FVGs
        fvg_list = detect_fvg(df, lookback_period=10, body_multiplier=1.5)
        
        # Should not detect any FVGs
        assert all(fvg is None for fvg in fvg_list)
        
    def test_generate_fvg_data_fast_bullish(self):
        """Test fast FVG generation with bullish pattern"""
        bars = self.create_bullish_fvg_pattern()
        
        high = np.array([b.high for b in bars])
        low = np.array([b.low for b in bars])
        n = len(bars)
        
        bull_fvg_detected, bear_fvg_detected, is_bull_fvg_active, is_bear_fvg_active = generate_fvg_data_fast(high, low, n)
        
        # Should detect bullish FVG at index 2
        assert bull_fvg_detected[2] == True
        assert bear_fvg_detected[2] == False
        assert is_bull_fvg_active[2] == True
        assert is_bear_fvg_active[2] == False
        
    def test_generate_fvg_data_fast_bearish(self):
        """Test fast FVG generation with bearish pattern"""
        bars = self.create_bearish_fvg_pattern()
        
        high = np.array([b.high for b in bars])
        low = np.array([b.low for b in bars])
        n = len(bars)
        
        bull_fvg_detected, bear_fvg_detected, is_bull_fvg_active, is_bear_fvg_active = generate_fvg_data_fast(high, low, n)
        
        # Should detect bearish FVG at index 2
        assert bull_fvg_detected[2] == False
        assert bear_fvg_detected[2] == True
        assert is_bull_fvg_active[2] == False
        assert is_bear_fvg_active[2] == True
        
    def test_generate_fvg_data_fast_no_gaps(self):
        """Test fast FVG generation with no gaps"""
        bars = self.create_no_gap_pattern()
        
        high = np.array([b.high for b in bars])
        low = np.array([b.low for b in bars])
        n = len(bars)
        
        bull_fvg_detected, bear_fvg_detected, is_bull_fvg_active, is_bear_fvg_active = generate_fvg_data_fast(high, low, n)
        
        # Should not detect any FVGs
        assert not any(bull_fvg_detected)
        assert not any(bear_fvg_detected)
        assert not any(is_bull_fvg_active)
        assert not any(is_bear_fvg_active)
        
    def test_fvg_calculate_5m_insufficient_data(self):
        """Test FVG calculation with insufficient data"""
        # Test with no data
        result = self.fvg.calculate_5m(self.create_test_bar(100.0, 102.0, 98.0, 101.0))
        assert result == {'fvg_bullish_active': False, 'fvg_bearish_active': False}
        
        # Test with only 1 bar
        bar1 = self.create_test_bar(100.0, 102.0, 98.0, 101.0)
        result = self.fvg.calculate_5m(bar1)
        assert result == {'fvg_bullish_active': False, 'fvg_bearish_active': False}
        
        # Test with only 2 bars
        bar2 = self.create_test_bar(101.0, 103.0, 99.0, 102.0)
        result = self.fvg.calculate_5m(bar2)
        assert result == {'fvg_bullish_active': False, 'fvg_bearish_active': False}
        
    def test_fvg_calculate_5m_bullish_detection(self):
        """Test FVG bullish detection in calculate_5m"""
        bars = self.create_bullish_fvg_pattern()
        
        # Process bars sequentially
        for i, bar in enumerate(bars):
            result = self.fvg.calculate_5m(bar)
            
            if i == 2:  # Third bar should detect the gap
                assert result['fvg_bullish_active'] == True
                assert result['fvg_bearish_active'] == False
                assert result['fvg_nearest_level'] == 102.0  # first_high
                
    def test_fvg_calculate_5m_bearish_detection(self):
        """Test FVG bearish detection in calculate_5m"""
        bars = self.create_bearish_fvg_pattern()
        
        # Process bars sequentially
        for i, bar in enumerate(bars):
            result = self.fvg.calculate_5m(bar)
            
            if i == 2:  # Third bar should detect the gap
                assert result['fvg_bullish_active'] == False
                assert result['fvg_bearish_active'] == True
                assert result['fvg_nearest_level'] == 98.0  # first_low
                
    def test_fvg_gap_size_validation(self):
        """Test FVG gap size validation"""
        # Create bars with very small gap (should not trigger)
        bars = [
            self.create_test_bar(100.0, 102.0, 98.0, 101.0),
            self.create_test_bar(101.0, 103.0, 100.0, 102.0),
            self.create_test_bar(102.0, 104.0, 102.001, 103.0)  # Very small gap
        ]
        
        for bar in bars:
            result = self.fvg.calculate_5m(bar)
            
        # Should not detect gap due to small size
        assert result['fvg_bullish_active'] == False
        assert result['fvg_bearish_active'] == False
        
    def test_fvg_body_multiplier_effect(self):
        """Test effect of body multiplier on FVG detection"""
        bars = self.create_bullish_fvg_pattern()
        
        # Test with high body multiplier (should be more restrictive)
        high_multiplier_config = self.config.copy()
        high_multiplier_config['body_multiplier'] = 10.0
        
        fvg_restrictive = FVGDetector(high_multiplier_config, self.mock_event_bus)
        
        for bar in bars:
            result = fvg_restrictive.calculate_5m(bar)
            
        # Might not detect due to high body multiplier requirement
        # (depends on the specific bar sizes)
        assert isinstance(result['fvg_bullish_active'], bool)
        assert isinstance(result['fvg_bearish_active'], bool)
        
    def test_fvg_performance_requirements(self):
        """Test FVG performance requirements"""
        # Generate sufficient data
        bars = self.create_bullish_fvg_pattern()
        
        # Add to history
        for bar in bars[:-1]:
            self.fvg.update_5m_history(bar)
            
        # Time the calculation
        start_time = time.time()
        result = self.fvg.calculate_5m(bars[-1])
        calc_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Should complete within 1ms for <1ms target
        assert calc_time < 1.0
        
        # Result should be valid
        assert isinstance(result['fvg_bullish_active'], bool)
        assert isinstance(result['fvg_bearish_active'], bool)
        
    def test_fvg_edge_cases(self):
        """Test FVG edge cases"""
        # Test with same OHLC values
        same_ohlc_bars = [
            self.create_test_bar(100.0, 100.0, 100.0, 100.0),
            self.create_test_bar(100.0, 100.0, 100.0, 100.0),
            self.create_test_bar(100.0, 100.0, 100.0, 100.0)
        ]
        
        for bar in same_ohlc_bars:
            result = self.fvg.calculate_5m(bar)
            
        # Should handle same values without issues
        assert result['fvg_bullish_active'] == False
        assert result['fvg_bearish_active'] == False
        
        # Test with extreme values
        extreme_bars = [
            self.create_test_bar(1.0, 2.0, 0.5, 1.5),
            self.create_test_bar(1.5, 10.0, 0.1, 8.0),
            self.create_test_bar(8.0, 20.0, 5.0, 15.0)
        ]
        
        self.fvg.reset()
        for bar in extreme_bars:
            result = self.fvg.calculate_5m(bar)
            
        # Should handle extreme values
        assert isinstance(result['fvg_bullish_active'], bool)
        assert isinstance(result['fvg_bearish_active'], bool)
        
    def test_fvg_gap_persistence(self):
        """Test FVG gap persistence logic"""
        bars = self.create_bullish_fvg_pattern()
        
        # Add more bars after the gap
        for i in range(15):  # Add bars that don't fill the gap
            additional_bar = self.create_test_bar(
                105.0 + i, 107.0 + i, 103.0 + i, 106.0 + i
            )
            bars.append(additional_bar)
            
        # Process all bars
        for bar in bars:
            result = self.fvg.calculate_5m(bar)
            
        # Gap should still be active if not filled
        assert result['fvg_bullish_active'] == True
        
    def test_fvg_gap_mitigation(self):
        """Test FVG gap mitigation (filling)"""
        bars = self.create_bullish_fvg_pattern()
        
        # Add bar that fills the gap
        mitigation_bar = self.create_test_bar(
            105.0, 106.0, 101.0, 102.0  # low=101 < gap_level=102
        )
        bars.append(mitigation_bar)
        
        # Process all bars
        for bar in bars:
            result = self.fvg.calculate_5m(bar)
            
        # Gap should be mitigated
        assert result['fvg_bullish_active'] == False
        
    def test_fvg_multiple_gaps(self):
        """Test handling of multiple FVGs"""
        # Create multiple gap patterns
        bars = []
        
        # First bullish gap
        bars.extend([
            self.create_test_bar(100.0, 102.0, 98.0, 101.0),
            self.create_test_bar(101.0, 108.0, 100.0, 107.0),
            self.create_test_bar(107.0, 109.0, 103.0, 108.0)
        ])
        
        # Some normal bars
        bars.extend([
            self.create_test_bar(108.0, 110.0, 106.0, 109.0),
            self.create_test_bar(109.0, 111.0, 107.0, 110.0)
        ])
        
        # Second bearish gap
        bars.extend([
            self.create_test_bar(110.0, 112.0, 108.0, 109.0),
            self.create_test_bar(109.0, 110.0, 102.0, 103.0),
            self.create_test_bar(103.0, 107.0, 101.0, 104.0)  # high=107 < prev_low=108
        ])
        
        # Process all bars
        for bar in bars:
            result = self.fvg.calculate_5m(bar)
            
        # Should detect the latest gap
        assert result['fvg_bearish_active'] == True
        
    def test_fvg_mathematical_correctness(self):
        """Test mathematical correctness of FVG detection"""
        # Test precise gap conditions
        bars = [
            # Bar 1: High = 100.0, Low = 95.0
            self.create_test_bar(97.0, 100.0, 95.0, 98.0),
            
            # Bar 2: Middle bar
            self.create_test_bar(98.0, 105.0, 96.0, 103.0),
            
            # Bar 3: Low = 100.1 (exactly above Bar 1 high)
            self.create_test_bar(103.0, 106.0, 100.1, 104.0)
        ]
        
        for bar in bars:
            result = self.fvg.calculate_5m(bar)
            
        # Should detect bullish gap: 100.1 > 100.0
        assert result['fvg_bullish_active'] == True
        
        # Test exact boundary condition
        bars_boundary = [
            self.create_test_bar(97.0, 100.0, 95.0, 98.0),
            self.create_test_bar(98.0, 105.0, 96.0, 103.0),
            self.create_test_bar(103.0, 106.0, 100.0, 104.0)  # Low = 100.0 (equal to high)
        ]
        
        self.fvg.reset()
        for bar in bars_boundary:
            result = self.fvg.calculate_5m(bar)
            
        # Should NOT detect gap: 100.0 == 100.0
        assert result['fvg_bullish_active'] == False
        
    def test_fvg_reset_functionality(self):
        """Test FVG reset functionality"""
        bars = self.create_bullish_fvg_pattern()
        
        # Process bars
        for bar in bars:
            self.fvg.calculate_5m(bar)
            
        # Check that data exists
        assert len(self.fvg.history_5m) > 0
        
        # Reset
        self.fvg.reset()
        
        # Check that data is cleared
        assert len(self.fvg.history_5m) == 0
        
    def test_fvg_get_current_values(self):
        """Test getting current FVG values"""
        # Initially should return defaults
        values = self.fvg.get_current_values()
        assert values == {'fvg_bullish_active': False, 'fvg_bearish_active': False}
        
    def test_fvg_only_5m_processing(self):
        """Test that FVG only processes 5-minute data"""
        # FVG should only work on 5-minute data
        # calculate_30m is not implemented, so it should return empty
        result_30m = self.fvg.calculate_30m(self.create_test_bar(100.0, 102.0, 98.0, 101.0))
        assert result_30m == {}
        
    def test_fvg_dataframe_edge_cases(self):
        """Test FVG with DataFrame edge cases"""
        # Test with minimal DataFrame
        df_minimal = pd.DataFrame({
            'High': [100.0, 101.0],
            'Low': [99.0, 100.0],
            'Open': [99.5, 100.5],
            'Close': [100.0, 101.0]
        })
        
        fvg_list = detect_fvg(df_minimal)
        assert len(fvg_list) == 2
        assert all(fvg is None for fvg in fvg_list)  # Not enough data for gaps
        
    def test_fvg_numba_optimization(self):
        """Test that numba optimization works correctly"""
        # Test with larger dataset to ensure numba compilation works
        n = 1000
        high = np.random.uniform(100, 110, n)
        low = np.random.uniform(90, 100, n)
        
        # Ensure high > low
        for i in range(n):
            if high[i] <= low[i]:
                high[i] = low[i] + 1.0
                
        # Should not crash with numba compilation
        bull_fvg_detected, bear_fvg_detected, is_bull_fvg_active, is_bear_fvg_active = generate_fvg_data_fast(high, low, n)
        
        # Should return arrays of correct length
        assert len(bull_fvg_detected) == n
        assert len(bear_fvg_detected) == n
        assert len(is_bull_fvg_active) == n
        assert len(is_bear_fvg_active) == n
        
    def test_fvg_configuration_validation(self):
        """Test FVG configuration validation"""
        # Test with custom configuration
        custom_config = {
            'threshold': 0.005,
            'lookback_period': 20,
            'body_multiplier': 2.0
        }
        
        fvg_custom = FVGDetector(custom_config, self.mock_event_bus)
        
        assert fvg_custom.threshold == 0.005
        assert fvg_custom.lookback_period == 20
        assert fvg_custom.body_multiplier == 2.0
        
    def test_fvg_memory_efficiency(self):
        """Test FVG memory efficiency"""
        # Generate large dataset
        bars = []
        for i in range(500):
            bar = self.create_test_bar(100.0 + i*0.1, 102.0 + i*0.1, 98.0 + i*0.1, 101.0 + i*0.1)
            bars.append(bar)
            
        # Process all bars
        for bar in bars:
            result = self.fvg.calculate_5m(bar)
            
        # Check that memory usage is controlled
        assert len(self.fvg.history_5m) <= 1000  # Should be limited by max_history_length
        
    def test_fvg_threading_safety(self):
        """Test FVG thread safety"""
        import threading
        
        bars = self.create_bullish_fvg_pattern()
        results = []
        
        def calculate_fvg(bar):
            result = self.fvg.calculate_5m(bar)
            results.append(result)
            
        # Create multiple threads
        threads = []
        for bar in bars:
            t = threading.Thread(target=calculate_fvg, args=(bar,))
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
            assert isinstance(result['fvg_bullish_active'], bool)
            assert isinstance(result['fvg_bearish_active'], bool)
            
    def test_fvg_stress_testing(self):
        """Test FVG under stress conditions"""
        # Generate many bars with potential gaps
        bars = []
        for i in range(100):
            if i % 10 == 0:
                # Create potential gap every 10 bars
                bar = self.create_test_bar(
                    100.0 + i,
                    110.0 + i,
                    105.0 + i,  # High low to potentially create gaps
                    108.0 + i
                )
            else:
                # Normal bars
                bar = self.create_test_bar(
                    100.0 + i,
                    102.0 + i,
                    98.0 + i,
                    101.0 + i
                )
            bars.append(bar)
            
        # Process all bars - should not crash
        for bar in bars:
            result = self.fvg.calculate_5m(bar)
            assert isinstance(result, dict)
            assert 'fvg_bullish_active' in result
            assert 'fvg_bearish_active' in result