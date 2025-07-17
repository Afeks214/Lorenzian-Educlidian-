"""
Mathematical Validation Tests for LVN (Low Volume Node) Analyzer
Tests market profile calculations, LVN identification, and performance requirements
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import time

from src.indicators.custom.lvn import (
    LVNAnalyzer, 
    identify_and_classify_lvns,
    calculate_lvn_characteristics,
    find_post_lvn_range,
    MarketProfile
)
from src.core.minimal_dependencies import EventBus, BarData
from tests.mocks.mock_event_bus import MockEventBus


class TestLVNAnalyzer:
    """Test suite for LVN Analyzer mathematical validation"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.mock_event_bus = MockEventBus()
        self.config = {
            'lookback_periods': 20,
            'strength_threshold': 0.7,
            'max_history_length': 100
        }
        self.lvn = LVNAnalyzer(self.config, self.mock_event_bus)
        
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
    
    def create_volume_profile_data(self, length=50, create_lvn=True):
        """Create test data with known volume profile characteristics"""
        bars = []
        base_price = 100.0
        
        for i in range(length):
            # Create varying volume patterns
            if create_lvn and 20 <= i <= 30:
                # Low volume area (potential LVN)
                volume = np.random.uniform(100, 300)
                price = base_price + np.random.uniform(-0.5, 0.5)
            elif 10 <= i <= 15:
                # High volume area (POC region)
                volume = np.random.uniform(2000, 5000)
                price = base_price + np.random.uniform(-1.0, 1.0)
            else:
                # Normal volume
                volume = np.random.uniform(500, 1500)
                price = base_price + np.random.uniform(-2.0, 2.0) + (i * 0.1)
            
            timestamp = datetime.now() + timedelta(minutes=i*30)
            bar = self.create_test_bar(
                close=price,
                high=price + np.random.uniform(0.1, 1.0),
                low=price - np.random.uniform(0.1, 1.0),
                volume=int(volume),
                timestamp=timestamp
            )
            bars.append(bar)
            
        return bars
    
    def test_lvn_initialization(self):
        """Test LVN analyzer initialization"""
        assert self.lvn.lookback_periods == 20
        assert self.lvn.strength_threshold == 0.7
        assert self.lvn.max_history_length == 100
        assert len(self.lvn.history_buffer) == 0
        assert len(self.lvn.current_lvns) == 0
        assert self.lvn.last_calculation_time is None
        
    def test_market_profile_basic(self):
        """Test basic MarketProfile functionality"""
        # Create simple test data
        df = pd.DataFrame({
            'High': [102.0, 103.0, 104.0],
            'Low': [98.0, 99.0, 100.0],
            'Volume': [1000, 1500, 2000]
        })
        
        mp = MarketProfile(df, tick_size=0.25)
        
        # Test profile calculation
        profile = mp.profile
        assert isinstance(profile, pd.Series)
        assert not profile.empty
        assert all(profile.values >= 0)  # Volume should be non-negative
        
        # Test POC calculation
        poc = mp.poc_price
        assert isinstance(poc, (int, float))
        assert poc > 0
        
        # Test value area calculation
        value_area = mp.value_area
        assert isinstance(value_area, tuple)
        assert len(value_area) == 2
        
    def test_market_profile_edge_cases(self):
        """Test MarketProfile with edge cases"""
        # Test with empty DataFrame
        df_empty = pd.DataFrame()
        mp_empty = MarketProfile(df_empty)
        
        profile_empty = mp_empty.profile
        assert profile_empty.empty
        
        poc_empty = mp_empty.poc_price
        assert poc_empty == 0
        
        value_area_empty = mp_empty.value_area
        assert value_area_empty == (0, 0)
        
        # Test with single bar
        df_single = pd.DataFrame({
            'High': [102.0],
            'Low': [98.0],
            'Volume': [1000]
        })
        
        mp_single = MarketProfile(df_single)
        profile_single = mp_single.profile
        assert not profile_single.empty
        
    def test_find_post_lvn_range_numba(self):
        """Test Numba-accelerated post-LVN range finding"""
        # Test with normal data
        prices = np.array([100.0, 101.0, 102.0, 99.0, 98.0])
        min_price, max_price = find_post_lvn_range(prices)
        
        assert min_price == 98.0
        assert max_price == 102.0
        
        # Test with empty array
        empty_prices = np.array([])
        min_empty, max_empty = find_post_lvn_range(empty_prices)
        
        assert np.isnan(min_empty)
        assert np.isnan(max_empty)
        
        # Test with single element
        single_price = np.array([100.0])
        min_single, max_single = find_post_lvn_range(single_price)
        
        assert min_single == 100.0
        assert max_single == 100.0
        
    def test_lvn_calculation_insufficient_data(self):
        """Test LVN calculation with insufficient data"""
        # Test with no data
        result = self.lvn.calculate_30m(self.create_test_bar())
        expected = {
            'nearest_lvn_price': 0.0,
            'nearest_lvn_strength': 0.0,
            'distance_to_nearest_lvn': 0.0
        }
        assert result == expected
        
        # Test with insufficient data (< lookback_periods)
        for i in range(10):  # Less than lookback_periods (20)
            bar = self.create_test_bar(close=100.0 + i)
            result = self.lvn.calculate_30m(bar)
            assert result == expected
            
    def test_lvn_calculation_sufficient_data(self):
        """Test LVN calculation with sufficient data"""
        # Generate test data with known volume profile
        bars = self.create_volume_profile_data(30, create_lvn=True)
        
        # Process bars
        for bar in bars:
            result = self.lvn.calculate_30m(bar)
            
        # Final result should be valid
        assert isinstance(result['nearest_lvn_price'], float)
        assert isinstance(result['nearest_lvn_strength'], float)
        assert isinstance(result['distance_to_nearest_lvn'], float)
        assert result['nearest_lvn_strength'] >= 0.0
        assert result['nearest_lvn_strength'] <= 1.0
        assert result['distance_to_nearest_lvn'] >= 0.0
        
    def test_lvn_strength_calculation(self):
        """Test LVN strength calculation logic"""
        # Create data with clear low volume areas
        bars = []
        base_price = 100.0
        
        for i in range(25):
            if 10 <= i <= 15:
                # High volume area (POC)
                volume = 5000
                price = base_price
            elif i == 20:
                # Low volume area (LVN)
                volume = 500
                price = base_price + 5.0
            else:
                # Normal volume
                volume = 2000
                price = base_price + (i * 0.1)
            
            timestamp = datetime.now() + timedelta(minutes=i*30)
            bar = self.create_test_bar(
                close=price,
                volume=volume,
                timestamp=timestamp
            )
            bars.append(bar)
            
        # Process bars
        for bar in bars:
            result = self.lvn.calculate_30m(bar)
            
        # Should detect the LVN and calculate strength
        assert result['nearest_lvn_strength'] > 0.0
        # With 500 volume vs 5000 POC, strength should be high
        assert result['nearest_lvn_strength'] > 0.5
        
    def test_lvn_distance_calculation(self):
        """Test LVN distance calculation"""
        # Create data with known LVN location
        bars = []
        lvn_price = 105.0
        current_price = 100.0
        
        for i in range(25):
            if i == 15:
                # Create LVN at specific price
                volume = 200
                price = lvn_price
            else:
                # Normal volume and prices
                volume = 1000
                price = current_price + (i * 0.1)
            
            timestamp = datetime.now() + timedelta(minutes=i*30)
            bar = self.create_test_bar(
                close=price,
                volume=volume,
                timestamp=timestamp
            )
            bars.append(bar)
            
        # Process bars
        for bar in bars:
            result = self.lvn.calculate_30m(bar)
            
        # Check distance calculation
        final_price = bars[-1].close
        expected_distance = abs(result['nearest_lvn_price'] - final_price)
        assert abs(result['distance_to_nearest_lvn'] - expected_distance) < 0.1
        
    def test_lvn_performance_requirements(self):
        """Test LVN performance requirements"""
        # Generate sufficient data
        bars = self.create_volume_profile_data(25)
        
        # Add to history
        for bar in bars[:-1]:
            self.lvn.calculate_30m(bar)
            
        # Time the calculation
        start_time = time.time()
        result = self.lvn.calculate_30m(bars[-1])
        calc_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Should complete within 1ms for <1ms target
        assert calc_time < 1.0
        
        # Result should be valid
        assert isinstance(result['nearest_lvn_price'], float)
        assert isinstance(result['nearest_lvn_strength'], float)
        assert isinstance(result['distance_to_nearest_lvn'], float)
        
    def test_lvn_buffer_management(self):
        """Test LVN history buffer management"""
        # Generate more data than max_history_length
        bars = self.create_volume_profile_data(150)  # More than max_history_length (100)
        
        # Process all bars
        for bar in bars:
            self.lvn.calculate_30m(bar)
            
        # Check that buffer is limited
        assert len(self.lvn.history_buffer) <= self.lvn.max_history_length
        
        # Should contain the most recent data
        latest_bar = bars[-1]
        buffer_latest = self.lvn.history_buffer[-1]
        assert buffer_latest['timestamp'] == latest_bar.timestamp
        assert buffer_latest['close'] == latest_bar.close
        
    def test_lvn_mathematical_properties(self):
        """Test mathematical properties of LVN calculations"""
        # Generate test data
        bars = self.create_volume_profile_data(50)
        
        results = []
        for bar in bars:
            result = self.lvn.calculate_30m(bar)
            results.append(result)
            
        # Check mathematical properties
        valid_results = [r for r in results if r['nearest_lvn_strength'] > 0]
        
        for result in valid_results:
            # Strength should be between 0 and 1
            assert 0.0 <= result['nearest_lvn_strength'] <= 1.0
            
            # Distance should be non-negative
            assert result['distance_to_nearest_lvn'] >= 0.0
            
            # Price should be positive
            assert result['nearest_lvn_price'] > 0.0
            
    def test_lvn_edge_cases(self):
        """Test LVN edge cases"""
        # Test with all same volumes
        same_volume_bars = []
        for i in range(25):
            bar = self.create_test_bar(
                close=100.0 + i,
                volume=1000,  # Same volume for all bars
                timestamp=datetime.now() + timedelta(minutes=i*30)
            )
            same_volume_bars.append(bar)
            
        # Process bars
        for bar in same_volume_bars:
            result = self.lvn.calculate_30m(bar)
            
        # Should handle same volumes gracefully
        assert isinstance(result['nearest_lvn_price'], float)
        assert isinstance(result['nearest_lvn_strength'], float)
        assert isinstance(result['distance_to_nearest_lvn'], float)
        
        # Test with extreme volumes
        extreme_bars = []
        for i in range(25):
            volume = 1 if i == 10 else 1000000  # Very low vs very high
            bar = self.create_test_bar(
                close=100.0 + i,
                volume=volume,
                timestamp=datetime.now() + timedelta(minutes=i*30)
            )
            extreme_bars.append(bar)
            
        # Reset for new test
        self.lvn = LVNAnalyzer(self.config, self.mock_event_bus)
        
        for bar in extreme_bars:
            result = self.lvn.calculate_30m(bar)
            
        # Should handle extreme volumes
        assert isinstance(result['nearest_lvn_price'], float)
        assert isinstance(result['nearest_lvn_strength'], float)
        
    def test_lvn_default_result(self):
        """Test LVN default result handling"""
        default_result = self.lvn._default_result()
        
        expected = {
            'nearest_lvn_price': 0.0,
            'nearest_lvn_strength': 0.0,
            'distance_to_nearest_lvn': 0.0
        }
        
        assert default_result == expected
        
    def test_lvn_error_handling(self):
        """Test LVN error handling"""
        # Test with invalid bar data
        invalid_bar = BarData(
            symbol="BTCUSDT",
            timestamp=datetime.now(),
            open=float('nan'),
            high=float('nan'),
            low=float('nan'),
            close=float('nan'),
            volume=0,
            timeframe=30
        )
        
        # Should not crash and return default result
        result = self.lvn.calculate_30m(invalid_bar)
        assert result == self.lvn._default_result()
        
    def test_identify_and_classify_lvns(self):
        """Test LVN identification and classification"""
        # Create test DataFrame with datetime index
        dates = pd.date_range('2023-01-01', periods=100, freq='30T')
        df = pd.DataFrame({
            'Open': np.random.uniform(99, 101, 100),
            'High': np.random.uniform(101, 103, 100),
            'Low': np.random.uniform(97, 99, 100),
            'Close': np.random.uniform(98, 102, 100),
            'Volume': np.random.uniform(500, 2000, 100)
        }, index=dates)
        
        # Test LVN identification
        lvn_df = identify_and_classify_lvns(df)
        
        # Should return a DataFrame
        assert isinstance(lvn_df, pd.DataFrame)
        
        # If LVNs found, should have required columns
        if not lvn_df.empty:
            assert 'lvn_price' in lvn_df.columns
            assert 'outcome' in lvn_df.columns
            
    def test_calculate_lvn_characteristics(self):
        """Test LVN characteristics calculation"""
        # Create test data
        dates = pd.date_range('2023-01-01', periods=50, freq='30T')
        df = pd.DataFrame({
            'Open': np.random.uniform(99, 101, 50),
            'High': np.random.uniform(101, 103, 50),
            'Low': np.random.uniform(97, 99, 50),
            'Close': np.random.uniform(98, 102, 50),
            'Volume': np.random.uniform(500, 2000, 50)
        }, index=dates)
        
        # Create mock LVN DataFrame
        lvn_df = pd.DataFrame({
            'lvn_price': [100.0, 101.0],
            'outcome': [0, 1]
        }, index=dates[:2])
        
        # Test characteristics calculation
        enhanced_df = calculate_lvn_characteristics(lvn_df, df)
        
        # Should return DataFrame with characteristics
        assert isinstance(enhanced_df, pd.DataFrame)
        
    def test_lvn_configuration_validation(self):
        """Test LVN configuration validation"""
        # Test with custom configuration
        custom_config = {
            'lookback_periods': 30,
            'strength_threshold': 0.8,
            'max_history_length': 200
        }
        
        lvn_custom = LVNAnalyzer(custom_config, self.mock_event_bus)
        
        assert lvn_custom.lookback_periods == 30
        assert lvn_custom.strength_threshold == 0.8
        assert lvn_custom.max_history_length == 200
        
    def test_lvn_memory_efficiency(self):
        """Test LVN memory efficiency"""
        # Generate large dataset
        bars = self.create_volume_profile_data(500)
        
        # Process all bars
        for bar in bars:
            result = self.lvn.calculate_30m(bar)
            
        # Check that memory usage is controlled
        assert len(self.lvn.history_buffer) <= self.lvn.max_history_length
        
    def test_lvn_numerical_stability(self):
        """Test LVN numerical stability"""
        # Test with very small volumes
        bars = []
        for i in range(25):
            volume = 0.001 if i == 10 else 1.0  # Very small vs small
            bar = self.create_test_bar(
                close=100.0 + i,
                volume=volume,
                timestamp=datetime.now() + timedelta(minutes=i*30)
            )
            bars.append(bar)
            
        # Should handle small volumes without numerical issues
        for bar in bars:
            result = self.lvn.calculate_30m(bar)
            assert not np.isnan(result['nearest_lvn_price'])
            assert not np.isnan(result['nearest_lvn_strength'])
            assert not np.isnan(result['distance_to_nearest_lvn'])
            assert not np.isinf(result['nearest_lvn_price'])
            assert not np.isinf(result['nearest_lvn_strength'])
            assert not np.isinf(result['distance_to_nearest_lvn'])
            
    def test_lvn_strength_threshold_effect(self):
        """Test effect of strength threshold on LVN detection"""
        # Test with different strength thresholds
        configs = [
            {'lookback_periods': 20, 'strength_threshold': 0.5, 'max_history_length': 100},
            {'lookback_periods': 20, 'strength_threshold': 0.8, 'max_history_length': 100}
        ]
        
        bars = self.create_volume_profile_data(30)
        results = []
        
        for config in configs:
            lvn = LVNAnalyzer(config, self.mock_event_bus)
            
            for bar in bars:
                result = lvn.calculate_30m(bar)
                
            results.append(result)
            
        # Different thresholds might give different results
        assert isinstance(results[0]['nearest_lvn_strength'], float)
        assert isinstance(results[1]['nearest_lvn_strength'], float)
        
    def test_lvn_threading_safety(self):
        """Test LVN thread safety"""
        import threading
        
        bars = self.create_volume_profile_data(20)
        results = []
        
        def calculate_lvn(bar):
            result = self.lvn.calculate_30m(bar)
            results.append(result)
            
        # Create multiple threads
        threads = []
        for bar in bars:
            t = threading.Thread(target=calculate_lvn, args=(bar,))
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
            assert isinstance(result['nearest_lvn_price'], float)
            assert isinstance(result['nearest_lvn_strength'], float)
            assert isinstance(result['distance_to_nearest_lvn'], float)
            
    def test_lvn_dataframe_conversion(self):
        """Test DataFrame conversion logic"""
        # Add some bars to history
        bars = self.create_volume_profile_data(25)
        
        for bar in bars:
            self.lvn.calculate_30m(bar)
            
        # Check that history buffer contains expected data
        assert len(self.lvn.history_buffer) == 25
        
        # Convert to DataFrame (simulate internal conversion)
        df = pd.DataFrame(list(self.lvn.history_buffer))
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Should have correct columns
        expected_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in expected_columns:
            assert col in df.columns
            
        # Should have correct data types
        assert df['open'].dtype in [np.float64, np.int64]
        assert df['high'].dtype in [np.float64, np.int64]
        assert df['low'].dtype in [np.float64, np.int64]
        assert df['close'].dtype in [np.float64, np.int64]
        assert df['volume'].dtype in [np.float64, np.int64]
        
    def test_lvn_tick_size_effect(self):
        """Test effect of tick size on market profile"""
        # Test with different tick sizes
        df = pd.DataFrame({
            'High': [102.0, 103.0, 104.0],
            'Low': [98.0, 99.0, 100.0],
            'Volume': [1000, 1500, 2000]
        })
        
        tick_sizes = [0.1, 0.25, 0.5, 1.0]
        profiles = []
        
        for tick_size in tick_sizes:
            mp = MarketProfile(df, tick_size=tick_size)
            profile = mp.profile
            profiles.append(profile)
            
        # Different tick sizes should give different profile granularities
        assert len(profiles[0]) >= len(profiles[1])  # Smaller tick = more levels
        assert len(profiles[1]) >= len(profiles[2])
        assert len(profiles[2]) >= len(profiles[3])
        
    def test_lvn_stress_testing(self):
        """Test LVN under stress conditions"""
        # Generate large dataset with complex patterns
        bars = []
        for i in range(100):
            # Create complex volume patterns
            if i % 20 == 0:
                volume = np.random.uniform(50, 200)  # Low volume areas
            elif i % 15 == 0:
                volume = np.random.uniform(3000, 8000)  # High volume areas
            else:
                volume = np.random.uniform(500, 2000)  # Normal volume
                
            price = 100.0 + np.sin(i * 0.1) * 10 + np.random.normal(0, 0.5)
            
            bar = self.create_test_bar(
                close=price,
                volume=int(volume),
                timestamp=datetime.now() + timedelta(minutes=i*30)
            )
            bars.append(bar)
            
        # Process all bars - should not crash
        for bar in bars:
            result = self.lvn.calculate_30m(bar)
            assert isinstance(result, dict)
            assert 'nearest_lvn_price' in result
            assert 'nearest_lvn_strength' in result
            assert 'distance_to_nearest_lvn' in result