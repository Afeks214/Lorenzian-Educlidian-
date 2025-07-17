"""
Comprehensive Test Suite for Indicators Engine
Tests engine lifecycle, indicator registration, calculation pipeline, and performance
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
import time

from src.indicators.engine import IndicatorEngine
from src.core.kernel import AlgoSpaceKernel
from src.core.events import EventType, Event, BarData
from src.core.minimal_dependencies import EventBus
from tests.mocks.mock_kernel import MockKernel
from tests.mocks.mock_event_bus import MockEventBus


class TestIndicatorEngine:
    """Test suite for the IndicatorEngine component"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.mock_kernel = MockKernel()
        self.mock_event_bus = MockEventBus()
        self.engine = IndicatorEngine("test_engine", self.mock_kernel)
        
    def create_test_bar(self, symbol="BTCUSDT", timeframe=5, close=50000.0, timestamp=None):
        """Create a test bar for testing"""
        if timestamp is None:
            timestamp = datetime.now()
        
        return BarData(
            symbol=symbol,
            timestamp=timestamp,
            open=close - 10,
            high=close + 20,
            low=close - 30,
            close=close,
            volume=1000,
            timeframe=timeframe
        )
    
    def test_engine_initialization(self):
        """Test engine initialization and configuration"""
        assert self.engine.name == "test_engine"
        assert self.engine.kernel == self.mock_kernel
        assert isinstance(self.engine.feature_store, dict)
        assert len(self.engine.feature_store) > 0
        assert self.engine.calculations_5min == 0
        assert self.engine.calculations_30min == 0
        assert self.engine.events_emitted == 0
        
    def test_feature_store_initialization(self):
        """Test feature store initialization with default values"""
        expected_features = [
            'mlmi_value', 'mlmi_signal', 'nwrqk_value', 'nwrqk_slope', 'nwrqk_signal',
            'lvn_nearest_price', 'lvn_nearest_strength', 'lvn_distance_points',
            'fvg_bullish_active', 'fvg_bearish_active', 'fvg_nearest_level',
            'fvg_age', 'fvg_mitigation_signal', 'mmd_features',
            'mlmi_minus_nwrqk', 'mlmi_div_nwrqk', 'last_update_5min',
            'last_update_30min', 'calculation_status', 'feature_count'
        ]
        
        for feature in expected_features:
            assert feature in self.engine.feature_store
        
        # Check initial values
        assert self.engine.feature_store['mlmi_value'] == 0.0
        assert self.engine.feature_store['mlmi_signal'] == 0
        assert self.engine.feature_store['fvg_bullish_active'] == False
        assert self.engine.feature_store['fvg_bearish_active'] == False
        assert isinstance(self.engine.feature_store['mmd_features'], np.ndarray)
        assert len(self.engine.feature_store['mmd_features']) == 13
        
    def test_indicator_initialization(self):
        """Test that all indicators are properly initialized"""
        assert hasattr(self.engine, 'mlmi')
        assert hasattr(self.engine, 'nwrqk')
        assert hasattr(self.engine, 'fvg')
        assert hasattr(self.engine, 'lvn')
        assert hasattr(self.engine, 'mmd')
        
        # Check that indicators have correct default parameters
        assert self.engine.mlmi.num_neighbors == 200
        assert self.engine.nwrqk.h == 8.0
        assert self.engine.nwrqk.r == 8.0
        assert self.engine.fvg.threshold == 0.001
        
    def test_heiken_ashi_conversion(self):
        """Test Heiken Ashi conversion logic"""
        # Create sample bar data
        bar = self.create_test_bar(close=100.0)
        
        # Test first bar conversion
        ha_bar = self.engine._convert_to_heiken_ashi(bar)
        
        # HA Close = (O + H + L + C) / 4
        expected_ha_close = (bar.open + bar.high + bar.low + bar.close) / 4
        assert abs(ha_bar['close'] - expected_ha_close) < 0.001
        
        # For first bar: HA_Open = (Open + Close) / 2
        expected_ha_open = (bar.open + bar.close) / 2
        assert abs(ha_bar['open'] - expected_ha_open) < 0.001
        
        # Test subsequent bar conversion
        self.engine.ha_history_30m.append(ha_bar)
        bar2 = self.create_test_bar(close=105.0)
        ha_bar2 = self.engine._convert_to_heiken_ashi(bar2)
        
        # HA_Open = (Previous_HA_Open + Previous_HA_Close) / 2
        expected_ha_open2 = (ha_bar['open'] + ha_bar['close']) / 2
        assert abs(ha_bar2['open'] - expected_ha_open2) < 0.001
        
    def test_bar_data_validation(self):
        """Test bar data validation logic"""
        # Valid bar
        valid_bar = self.create_test_bar()
        assert self.engine._validate_bar_data(valid_bar, '5min') == True
        
        # Invalid symbol
        invalid_bar = self.create_test_bar(symbol="INVALID")
        assert self.engine._validate_bar_data(invalid_bar, '5min') == False
        
        # Invalid price data (negative prices)
        invalid_bar2 = BarData(
            symbol="BTCUSDT",
            timestamp=datetime.now(),
            open=-100,
            high=-50,
            low=-200,
            close=-100,
            volume=1000,
            timeframe=5
        )
        assert self.engine._validate_bar_data(invalid_bar2, '5min') == False
        
    def test_5min_bar_processing(self):
        """Test 5-minute bar processing workflow"""
        # Create test event
        bar = self.create_test_bar(timeframe=5)
        event = Event(EventType.NEW_5MIN_BAR, bar)
        
        # Process the event
        self.engine._on_5min_bar(event)
        
        # Check that bar was added to history
        assert len(self.engine.history_5m) == 1
        assert self.engine.history_5m[0] == bar
        
        # Check that calculations counter was incremented
        assert self.engine.calculations_5min == 1
        
    def test_30min_bar_processing(self):
        """Test 30-minute bar processing workflow"""
        # Create test event
        bar = self.create_test_bar(timeframe=30)
        event = Event(EventType.NEW_30MIN_BAR, bar)
        
        # Process the event
        self.engine._on_30min_bar(event)
        
        # Check that bar was added to history
        assert len(self.engine.history_30m) == 1
        assert self.engine.history_30m[0] == bar
        
        # Check that HA history was updated
        assert len(self.engine.ha_history_30m) == 1
        
        # Check that calculations counter was incremented
        assert self.engine.calculations_30min == 1
        assert self.engine.has_30min_data == True
        
    def test_fvg_tracking(self):
        """Test FVG tracking functionality"""
        # Create test bars for FVG detection
        bars = [
            self.create_test_bar(close=100.0, timestamp=datetime.now() - timedelta(minutes=10)),
            self.create_test_bar(close=105.0, timestamp=datetime.now() - timedelta(minutes=5)),
            self.create_test_bar(close=110.0, timestamp=datetime.now())
        ]
        
        # Add bars to history
        for bar in bars:
            self.engine.history_5m.append(bar)
        
        # Test FVG tracking update
        self.engine._update_fvg_tracking(bars)
        
        # Test finding nearest FVG
        nearest_fvg = self.engine._find_nearest_fvg(107.5)
        assert isinstance(nearest_fvg, dict)
        assert 'level' in nearest_fvg
        assert 'age' in nearest_fvg
        
    def test_lvn_strength_calculation(self):
        """Test LVN strength score calculation"""
        # Test with new LVN level
        strength = self.engine._calculate_lvn_strength_score(
            lvn_price=100.0,
            base_strength=0.7,
            current_price=100.1
        )
        
        # Should be close to base strength for new level
        assert 0.0 <= strength <= 1.0
        assert abs(strength - 0.7 * 0.4) < 0.1  # Base strength weighted by 0.4
        
        # Test with zero price
        strength_zero = self.engine._calculate_lvn_strength_score(
            lvn_price=0.0,
            base_strength=0.7,
            current_price=100.0
        )
        assert strength_zero == 0.0
        
    def test_interaction_features_calculation(self):
        """Test interaction features calculation"""
        # Set up test values
        self.engine.feature_store['mlmi_value'] = 0.5
        self.engine.feature_store['nwrqk_value'] = 0.3
        
        # Calculate interaction features
        self.engine._calculate_interaction_features()
        
        # Check results
        assert self.engine.feature_store['mlmi_minus_nwrqk'] == 0.2
        assert abs(self.engine.feature_store['mlmi_div_nwrqk'] - (0.5 / 0.3)) < 0.001
        
        # Test division by zero protection
        self.engine.feature_store['nwrqk_value'] = 0.0
        self.engine.feature_store['mlmi_value'] = 0.5
        self.engine._calculate_interaction_features()
        
        assert self.engine.feature_store['mlmi_div_nwrqk'] == 1.0
        
    def test_mmd_reference_distributions(self):
        """Test MMD reference distributions loading"""
        distributions = self.engine._load_reference_distributions()
        
        # Should have 7 reference distributions
        assert len(distributions) == 7
        
        # Each distribution should have correct shape
        for dist in distributions:
            assert dist.shape == (100, 4)  # 100 samples, 4 features
            
        # Check that distributions are different
        assert not np.array_equal(distributions[0], distributions[1])
        assert not np.array_equal(distributions[1], distributions[2])
        
    def test_history_buffer_limits(self):
        """Test that history buffers respect size limits"""
        # Fill buffers beyond limit
        for i in range(150):  # More than maxlen of 100
            bar = self.create_test_bar(close=100.0 + i)
            self.engine.history_5m.append(bar)
            self.engine.history_30m.append(bar)
            self.engine.ha_history_30m.append({
                'open': 100.0 + i,
                'high': 120.0 + i,
                'low': 80.0 + i,
                'close': 100.0 + i,
                'volume': 1000,
                'timestamp': datetime.now()
            })
        
        # Check that buffers are limited
        assert len(self.engine.history_5m) == 100
        assert len(self.engine.history_30m) == 100
        assert len(self.engine.ha_history_30m) == 100
        
    def test_get_current_features(self):
        """Test getting current feature store contents"""
        # Set some test values
        self.engine.feature_store['mlmi_value'] = 0.5
        self.engine.feature_store['fvg_bullish_active'] = True
        
        # Get current features
        features = self.engine.get_current_features()
        
        # Should be a deep copy
        assert features is not self.engine.feature_store
        assert features['mlmi_value'] == 0.5
        assert features['fvg_bullish_active'] == True
        
        # Modifying returned features shouldn't affect original
        features['mlmi_value'] = 0.7
        assert self.engine.feature_store['mlmi_value'] == 0.5
        
    def test_get_feature_summary(self):
        """Test feature summary generation"""
        summary = self.engine.get_feature_summary()
        
        expected_keys = [
            'total_features', 'calculations_5min', 'calculations_30min',
            'events_emitted', 'has_30min_data', 'last_update_5min',
            'last_update_30min', 'active_fvgs', 'history_sizes'
        ]
        
        for key in expected_keys:
            assert key in summary
            
        assert isinstance(summary['total_features'], int)
        assert isinstance(summary['calculations_5min'], int)
        assert isinstance(summary['calculations_30min'], int)
        assert isinstance(summary['events_emitted'], int)
        assert isinstance(summary['has_30min_data'], bool)
        assert isinstance(summary['active_fvgs'], int)
        assert isinstance(summary['history_sizes'], dict)
        
    @pytest.mark.asyncio
    async def test_async_start_stop(self):
        """Test async start and stop operations"""
        # Test start
        await self.engine.start()
        assert self.engine.is_running == True
        
        # Test stop
        await self.engine.stop()
        assert self.engine.is_running == False
        
    @pytest.mark.asyncio
    async def test_feature_store_atomic_updates(self):
        """Test atomic feature store updates"""
        # Test 5-minute update
        features_5m = {
            'fvg_bullish_active': True,
            'fvg_bearish_active': False,
            'fvg_nearest_level': 100.0
        }
        
        await self.engine._update_feature_store_5min(features_5m, datetime.now())
        
        assert self.engine.feature_store['fvg_bullish_active'] == True
        assert self.engine.feature_store['fvg_bearish_active'] == False
        assert self.engine.feature_store['fvg_nearest_level'] == 100.0
        
        # Test 30-minute update
        features_30m = {
            'mlmi_value': 0.5,
            'nwrqk_value': 0.3,
            'lvn_nearest_price': 100.0
        }
        
        await self.engine._update_feature_store_30min(features_30m, datetime.now())
        
        assert self.engine.feature_store['mlmi_value'] == 0.5
        assert self.engine.feature_store['nwrqk_value'] == 0.3
        assert self.engine.feature_store['lvn_nearest_price'] == 100.0
        
    def test_performance_timing(self):
        """Test that calculations meet performance requirements"""
        # Create sufficient bar data
        bars = [self.create_test_bar(close=100.0 + i) for i in range(100)]
        
        # Test 5-minute calculation performance
        start_time = time.time()
        features_5m = self.engine._calculate_5min_features(bars[-1])
        calc_time_5m = (time.time() - start_time) * 1000  # Convert to ms
        
        # Should be under 50ms for 5-minute calculations
        assert calc_time_5m < 50
        
        # Add bars to history for 30-minute calculations
        for bar in bars:
            self.engine.history_30m.append(bar)
            ha_bar = self.engine._convert_to_heiken_ashi(bar)
            self.engine.ha_history_30m.append(ha_bar)
            self.engine.volume_profile_buffer.append(bar)
        
        # Test 30-minute calculation performance
        start_time = time.time()
        features_30m = self.engine._calculate_30min_features(bars[-1], self.engine.ha_history_30m[-1])
        calc_time_30m = (time.time() - start_time) * 1000  # Convert to ms
        
        # Should be under 100ms for 30-minute calculations
        assert calc_time_30m < 100
        
    def test_error_handling(self):
        """Test error handling in various scenarios"""
        # Test with invalid bar data
        invalid_bar = None
        
        # Should not crash on invalid data
        try:
            self.engine._validate_bar_data(invalid_bar, '5min')
        except Exception:
            pass  # Expected to handle errors gracefully
        
        # Test with insufficient data
        features = self.engine._calculate_5min_features(self.create_test_bar())
        assert isinstance(features, dict)
        
        # Test with empty history
        self.engine.history_5m.clear()
        self.engine.history_30m.clear()
        features = self.engine._calculate_5min_features(self.create_test_bar())
        assert isinstance(features, dict)
        
    def test_memory_management(self):
        """Test memory management and cleanup"""
        # Fill history with large amount of data
        for i in range(1000):
            bar = self.create_test_bar(close=100.0 + i)
            self.engine.history_5m.append(bar)
            self.engine.history_30m.append(bar)
        
        # Check that deque limits are respected
        assert len(self.engine.history_5m) <= 100
        assert len(self.engine.history_30m) <= 100
        
        # Test LVN interaction history cleanup
        old_time = datetime.now() - timedelta(days=8)
        test_price = 100.0
        
        # Add old interaction
        self.engine.lvn_interaction_history[test_price] = [{
            'timestamp': old_time,
            'test_price': test_price,
            'approached_from': 'above',
            'volume': 1000
        }]
        
        # Calculate strength score (should trigger cleanup)
        strength = self.engine._calculate_lvn_strength_score(test_price, 0.7, test_price)
        
        # Old interaction should be cleaned up
        assert len(self.engine.lvn_interaction_history[test_price]) == 0
        
    def test_mathematical_correctness(self):
        """Test mathematical correctness of calculations"""
        # Test interaction features calculations
        self.engine.feature_store['mlmi_value'] = 0.6
        self.engine.feature_store['nwrqk_value'] = 0.4
        
        self.engine._calculate_interaction_features()
        
        # Check mathematical correctness
        assert abs(self.engine.feature_store['mlmi_minus_nwrqk'] - 0.2) < 1e-10
        assert abs(self.engine.feature_store['mlmi_div_nwrqk'] - 1.5) < 1e-10
        
        # Test edge cases
        self.engine.feature_store['nwrqk_value'] = 0.0
        self.engine.feature_store['mlmi_value'] = 0.0
        self.engine._calculate_interaction_features()
        
        assert self.engine.feature_store['mlmi_div_nwrqk'] == 0.0
        
    def test_concurrent_access(self):
        """Test concurrent access to feature store"""
        import threading
        
        def update_features():
            self.engine.feature_store['mlmi_value'] = 0.5
            time.sleep(0.001)
            self.engine.feature_store['nwrqk_value'] = 0.3
        
        def read_features():
            return self.engine.get_current_features()
        
        # Create multiple threads
        threads = []
        for i in range(10):
            t1 = threading.Thread(target=update_features)
            t2 = threading.Thread(target=read_features)
            threads.extend([t1, t2])
        
        # Start all threads
        for t in threads:
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        # Should not crash or corrupt data
        features = self.engine.get_current_features()
        assert isinstance(features, dict)
        
    def test_event_emission_conditions(self):
        """Test conditions for INDICATORS_READY event emission"""
        # Initially no 30-minute data
        assert self.engine.has_30min_data == False
        
        # Process 5-minute bar (should not emit event)
        bar_5m = self.create_test_bar(timeframe=5)
        event_5m = Event(EventType.NEW_5MIN_BAR, bar_5m)
        self.engine._on_5min_bar(event_5m)
        
        # Should not emit event yet
        assert self.engine.events_emitted == 0
        
        # Process 30-minute bar (should emit event)
        bar_30m = self.create_test_bar(timeframe=30)
        event_30m = Event(EventType.NEW_30MIN_BAR, bar_30m)
        self.engine._on_30min_bar(event_30m)
        
        # Should emit event now
        assert self.engine.events_emitted == 1
        assert self.engine.has_30min_data == True
        
    def test_configuration_validation(self):
        """Test configuration validation and defaults"""
        # Test with custom configuration
        custom_config = {
            'primary_symbol': 'ETHUSDT',
            'timeframes': ['5m', '30m'],
            'indicators': {
                'mlmi': {'num_neighbors': 300},
                'nwrqk': {'h': 10.0}
            }
        }
        
        # Create engine with custom config
        mock_kernel_custom = MockKernel(config=custom_config)
        engine_custom = IndicatorEngine("test_custom", mock_kernel_custom)
        
        # Check that custom parameters are used
        assert engine_custom.mlmi.num_neighbors == 300
        assert engine_custom.nwrqk.h == 10.0
        
    def test_data_flow_integrity(self):
        """Test data flow integrity through the engine"""
        # Create sequence of bars
        bars = []
        for i in range(50):
            bar = self.create_test_bar(
                close=100.0 + i,
                timestamp=datetime.now() + timedelta(minutes=i*5)
            )
            bars.append(bar)
        
        # Process bars in sequence
        for i, bar in enumerate(bars):
            if i < 25:  # First 25 as 5-minute bars
                event = Event(EventType.NEW_5MIN_BAR, bar)
                self.engine._on_5min_bar(event)
            else:  # Last 25 as 30-minute bars
                event = Event(EventType.NEW_30MIN_BAR, bar)
                self.engine._on_30min_bar(event)
        
        # Check data integrity
        assert len(self.engine.history_5m) == 25
        assert len(self.engine.history_30m) == 25
        assert len(self.engine.ha_history_30m) == 25
        
        # Check that latest bars are correct
        assert self.engine.history_5m[-1].close == 124.0
        assert self.engine.history_30m[-1].close == 149.0
        
        # Check calculations were performed
        assert self.engine.calculations_5min == 25
        assert self.engine.calculations_30min == 25
        
    def test_stress_testing(self):
        """Stress test the engine with high-frequency data"""
        import time
        
        # Generate large number of bars
        bars = [self.create_test_bar(close=100.0 + i) for i in range(1000)]
        
        # Measure processing time
        start_time = time.time()
        
        for bar in bars:
            event = Event(EventType.NEW_5MIN_BAR, bar)
            self.engine._on_5min_bar(event)
        
        processing_time = time.time() - start_time
        
        # Should handle 1000 bars in reasonable time (< 1 second)
        assert processing_time < 1.0
        
        # Check that all bars were processed
        assert self.engine.calculations_5min == 1000
        
        # Check memory usage is controlled
        assert len(self.engine.history_5m) == 100  # Limited by deque maxlen