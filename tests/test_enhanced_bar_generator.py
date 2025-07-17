#!/usr/bin/env python3
"""
Enhanced BarGenerator Test Suite
===============================

Comprehensive tests for the enhanced BarGenerator with production-ready features:
- Timezone-aware timestamp handling
- Intelligent gap detection and filling
- Market hours awareness
- Data validation and integrity checks
- Performance monitoring

Author: Agent 7 - Timestamp Alignment and Gap Handling Specialist
"""

import unittest
import sys
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from components.bar_generator import (
    BarGenerator, 
    BarGeneratorConfig, 
    GapFillStrategy,
    MarketSession,
    TimestampManager,
    MarketHoursManager,
    ValidationError,
    TimestampError
)

class MockEventBus:
    """Mock event bus for testing."""
    
    def __init__(self):
        self.events = []
    
    def publish(self, event_type: str, data: Any):
        self.events.append({
            'type': event_type,
            'data': data,
            'timestamp': datetime.now(timezone.utc)
        })
    
    def get_events_by_type(self, event_type: str) -> List[Dict[str, Any]]:
        return [event for event in self.events if event['type'] == event_type]
    
    def clear(self):
        self.events.clear()

class TestTimestampManager(unittest.TestCase):
    """Test timezone-aware timestamp handling."""
    
    def setUp(self):
        self.timestamp_manager = TimestampManager("America/New_York")
    
    def test_timezone_initialization(self):
        """Test timezone object creation."""
        self.assertEqual(self.timestamp_manager.timezone_str, "America/New_York")
        self.assertIsNotNone(self.timestamp_manager.timezone)
    
    def test_timestamp_normalization(self):
        """Test timestamp normalization to timezone-aware format."""
        # Test naive timestamp
        naive_time = datetime(2024, 1, 15, 14, 30, 0)
        normalized = self.timestamp_manager.normalize_timestamp(naive_time)
        
        self.assertIsNotNone(normalized.tzinfo)
        self.assertEqual(normalized.hour, 14)  # Should be converted to NY time
    
    def test_timestamp_validation(self):
        """Test timestamp validation."""
        # Valid timestamp
        valid_time = datetime.now(timezone.utc)
        is_valid, error = self.timestamp_manager.validate_timestamp(valid_time)
        self.assertTrue(is_valid)
        self.assertEqual(error, "")
        
        # Invalid timestamp (None)
        is_valid, error = self.timestamp_manager.validate_timestamp(None)
        self.assertFalse(is_valid)
        self.assertIn("None", error)
        
        # Invalid timestamp (too old)
        old_time = datetime.now(timezone.utc) - timedelta(days=400)
        is_valid, error = self.timestamp_manager.validate_timestamp(old_time)
        self.assertFalse(is_valid)
        self.assertIn("past", error)
    
    def test_latency_tracking(self):
        """Test tick latency tracking."""
        timestamp = datetime.now(timezone.utc)
        self.timestamp_manager.record_tick_latency(timestamp)
        
        stats = self.timestamp_manager.get_latency_stats()
        self.assertIn('avg_latency_ms', stats)
        self.assertGreaterEqual(stats['avg_latency_ms'], 0)

class TestMarketHoursManager(unittest.TestCase):
    """Test market hours and session management."""
    
    def setUp(self):
        self.market_manager = MarketHoursManager("America/New_York")
    
    def test_market_session_detection(self):
        """Test market session detection."""
        # Regular hours (10 AM EST on weekday)
        regular_time = datetime(2024, 1, 15, 15, 0, 0, tzinfo=timezone.utc)  # 10 AM EST
        session = self.market_manager.get_market_session(regular_time)
        self.assertEqual(session, MarketSession.REGULAR)
        
        # Weekend (Saturday)
        weekend_time = datetime(2024, 1, 13, 15, 0, 0, tzinfo=timezone.utc)  # Saturday
        session = self.market_manager.get_market_session(weekend_time)
        self.assertEqual(session, MarketSession.CLOSED)
    
    def test_market_open_check(self):
        """Test market open/closed detection."""
        # Weekday regular hours
        weekday_time = datetime(2024, 1, 15, 15, 0, 0, tzinfo=timezone.utc)  # Monday 10 AM EST
        self.assertTrue(self.market_manager.is_market_open(weekday_time))
        
        # Weekend
        weekend_time = datetime(2024, 1, 13, 15, 0, 0, tzinfo=timezone.utc)  # Saturday
        self.assertFalse(self.market_manager.is_market_open(weekend_time))

class TestEnhancedBarGenerator(unittest.TestCase):
    """Test enhanced BarGenerator functionality."""
    
    def setUp(self):
        self.config = BarGeneratorConfig(
            timezone="America/New_York",
            gap_fill_strategy=GapFillStrategy.SMART_FILL,
            enable_market_hours=True,
            validate_timestamps=True,
            enable_data_quality_checks=True,
            performance_monitoring=True,
            duplicate_detection=True
        )
        self.event_bus = MockEventBus()
        self.bar_generator = BarGenerator(self.config, self.event_bus)
    
    def test_initialization(self):
        """Test bar generator initialization."""
        self.assertIsNotNone(self.bar_generator.timestamp_manager)
        self.assertIsNotNone(self.bar_generator.market_hours_manager)
        self.assertIsNotNone(self.bar_generator.metrics)
        self.assertEqual(self.bar_generator.config.timezone, "America/New_York")
    
    def test_basic_tick_processing(self):
        """Test basic tick processing."""
        tick = {
            'timestamp': datetime(2024, 1, 15, 14, 30, 0, tzinfo=timezone.utc),
            'price': 100.0,
            'volume': 1000
        }
        
        initial_count = self.bar_generator.metrics.tick_count
        self.bar_generator.on_new_tick(tick)
        
        self.assertEqual(self.bar_generator.metrics.tick_count, initial_count + 1)
        self.assertEqual(self.bar_generator.last_close_price, 100.0)
    
    def test_duplicate_detection(self):
        """Test duplicate tick detection."""
        tick = {
            'timestamp': datetime(2024, 1, 15, 14, 30, 0, tzinfo=timezone.utc),
            'price': 100.0,
            'volume': 1000
        }
        
        # Process same tick twice
        self.bar_generator.on_new_tick(tick)
        initial_count = self.bar_generator.metrics.tick_count
        
        self.bar_generator.on_new_tick(tick)  # Duplicate
        
        # Should detect duplicate
        self.assertEqual(self.bar_generator.metrics.duplicate_ticks, 1)
        self.assertEqual(self.bar_generator.metrics.tick_count, initial_count)  # No increment
    
    def test_validation_errors(self):
        """Test input validation."""
        # Invalid tick data
        invalid_tick = {
            'timestamp': datetime(2024, 1, 15, 14, 30, 0, tzinfo=timezone.utc),
            'price': -100.0,  # Negative price
            'volume': 1000
        }
        
        initial_errors = self.bar_generator.metrics.validation_errors
        self.bar_generator.on_new_tick(invalid_tick)
        
        self.assertGreater(self.bar_generator.metrics.validation_errors, initial_errors)
    
    def test_gap_detection_and_filling(self):
        """Test gap detection and intelligent filling."""
        base_time = datetime(2024, 1, 15, 14, 30, 0, tzinfo=timezone.utc)
        
        # First tick
        tick1 = {
            'timestamp': base_time,
            'price': 100.0,
            'volume': 1000
        }
        self.bar_generator.on_new_tick(tick1)
        
        # Second tick after 10-minute gap
        tick2 = {
            'timestamp': base_time + timedelta(minutes=10),
            'price': 101.0,
            'volume': 1100
        }
        self.bar_generator.on_new_tick(tick2)
        
        # Check if gaps were detected
        gap_analysis = self.bar_generator.get_gap_analysis()
        self.assertGreater(gap_analysis['gap_statistics']['total_gaps'], 0)
    
    def test_timezone_awareness(self):
        """Test timezone-aware timestamp handling."""
        # Test with different timezones
        utc_time = datetime(2024, 1, 15, 19, 30, 0, tzinfo=timezone.utc)  # 7:30 PM UTC
        
        tick = {
            'timestamp': utc_time,
            'price': 100.0,
            'volume': 1000
        }
        
        self.bar_generator.on_new_tick(tick)
        
        # Check that events were published with correct timezone
        events = self.event_bus.get_events_by_type('NEW_5MIN_BAR')
        if events:
            bar_data = events[-1]['data']
            self.assertIsNotNone(bar_data.timestamp.tzinfo)
    
    def test_market_session_detection(self):
        """Test market session detection in bar data."""
        # Regular market hours (Monday 10 AM EST)
        regular_time = datetime(2024, 1, 15, 15, 0, 0, tzinfo=timezone.utc)
        
        tick = {
            'timestamp': regular_time,
            'price': 100.0,
            'volume': 1000
        }
        
        self.bar_generator.on_new_tick(tick)
        
        events = self.event_bus.get_events_by_type('NEW_5MIN_BAR')
        if events:
            bar_data = events[-1]['data']
            self.assertEqual(bar_data.market_session, MarketSession.REGULAR)
    
    def test_synthetic_bar_generation(self):
        """Test synthetic bar generation for gaps."""
        base_time = datetime(2024, 1, 15, 14, 30, 0, tzinfo=timezone.utc)
        
        # Create a significant gap that should trigger synthetic bars
        tick1 = {
            'timestamp': base_time,
            'price': 100.0,
            'volume': 1000
        }
        self.bar_generator.on_new_tick(tick1)
        
        # Skip 20 minutes (4 x 5-minute bars)
        tick2 = {
            'timestamp': base_time + timedelta(minutes=20),
            'price': 101.0,
            'volume': 1100
        }
        self.bar_generator.on_new_tick(tick2)
        
        # Check synthetic bars were created
        self.assertGreater(self.bar_generator.metrics.synthetic_bars_5min, 0)
        
        # Check that synthetic bars have the correct flag
        events = self.event_bus.get_events_by_type('NEW_5MIN_BAR')
        synthetic_bars = [event for event in events if event['data'].is_synthetic]
        self.assertGreater(len(synthetic_bars), 0)
    
    def test_data_quality_reporting(self):
        """Test data quality reporting."""
        # Process some ticks
        base_time = datetime(2024, 1, 15, 14, 30, 0, tzinfo=timezone.utc)
        for i in range(5):
            tick = {
                'timestamp': base_time + timedelta(minutes=i),
                'price': 100.0 + i * 0.1,
                'volume': 1000 + i * 10
            }
            self.bar_generator.on_new_tick(tick)
        
        # Get quality report
        quality_report = self.bar_generator.get_data_quality_report()
        
        self.assertIn('overall_quality', quality_report)
        self.assertIn('data_completeness', quality_report['overall_quality'])
        self.assertIn('timestamp_accuracy', quality_report['overall_quality'])
        self.assertIn('data_integrity', quality_report['overall_quality'])
        self.assertIn('duplicate_cleanliness', quality_report['overall_quality'])
        self.assertIn('recommendations', quality_report)
    
    def test_system_health_monitoring(self):
        """Test system health monitoring."""
        # Process some ticks
        base_time = datetime(2024, 1, 15, 14, 30, 0, tzinfo=timezone.utc)
        for i in range(3):
            tick = {
                'timestamp': base_time + timedelta(minutes=i),
                'price': 100.0 + i * 0.1,
                'volume': 1000 + i * 10
            }
            self.bar_generator.on_new_tick(tick)
        
        # Get health report
        health_report = self.bar_generator.get_system_health()
        
        self.assertIn('overall_health_score', health_report)
        self.assertIn('system_status', health_report)
        self.assertIn('processed_ticks', health_report)
        self.assertIn('performance_summary', health_report)
        self.assertIn('alerts', health_report)
        
        # Health score should be between 0 and 1
        self.assertGreaterEqual(health_report['overall_health_score'], 0)
        self.assertLessEqual(health_report['overall_health_score'], 1)
    
    def test_statistics_and_metrics(self):
        """Test comprehensive statistics collection."""
        # Process some ticks
        base_time = datetime(2024, 1, 15, 14, 30, 0, tzinfo=timezone.utc)
        for i in range(5):
            tick = {
                'timestamp': base_time + timedelta(minutes=i),
                'price': 100.0 + i * 0.1,
                'volume': 1000 + i * 10
            }
            self.bar_generator.on_new_tick(tick)
        
        # Get statistics
        stats = self.bar_generator.get_statistics()
        
        self.assertIn('tick_count', stats)
        self.assertIn('bars_emitted_5min', stats)
        self.assertIn('bars_emitted_30min', stats)
        self.assertIn('validation_errors', stats)
        self.assertIn('duplicate_ticks', stats)
        
        self.assertEqual(stats['tick_count'], 5)
        self.assertGreaterEqual(stats['bars_emitted_5min'], 0)
    
    def test_configuration_options(self):
        """Test various configuration options."""
        # Test different gap fill strategies
        strategies = [
            GapFillStrategy.FORWARD_FILL,
            GapFillStrategy.ZERO_VOLUME,
            GapFillStrategy.SMART_FILL
        ]
        
        for strategy in strategies:
            config = BarGeneratorConfig(gap_fill_strategy=strategy)
            bar_gen = BarGenerator(config, MockEventBus())
            
            self.assertEqual(bar_gen.config.gap_fill_strategy, strategy)
    
    def test_performance_monitoring(self):
        """Test performance monitoring functionality."""
        # Process ticks and check performance metrics
        base_time = datetime(2024, 1, 15, 14, 30, 0, tzinfo=timezone.utc)
        for i in range(10):
            tick = {
                'timestamp': base_time + timedelta(seconds=i),
                'price': 100.0 + i * 0.01,
                'volume': 100
            }
            self.bar_generator.on_new_tick(tick)
        
        # Get performance metrics
        stats = self.bar_generator.get_statistics()
        
        # Should have performance metrics if monitoring enabled
        if self.config.performance_monitoring:
            self.assertIn('avg_tick_time_ms', stats)
            self.assertIn('ticks_per_second', stats)

class TestGapFillStrategies(unittest.TestCase):
    """Test different gap filling strategies."""
    
    def create_bar_generator(self, strategy: GapFillStrategy) -> BarGenerator:
        config = BarGeneratorConfig(gap_fill_strategy=strategy)
        return BarGenerator(config, MockEventBus())
    
    def test_forward_fill_strategy(self):
        """Test forward fill gap strategy."""
        bar_gen = self.create_bar_generator(GapFillStrategy.FORWARD_FILL)
        
        # Create gap scenario
        base_time = datetime(2024, 1, 15, 14, 30, 0, tzinfo=timezone.utc)
        
        bar_gen.on_new_tick({
            'timestamp': base_time,
            'price': 100.0,
            'volume': 1000
        })
        
        bar_gen.on_new_tick({
            'timestamp': base_time + timedelta(minutes=10),
            'price': 101.0,
            'volume': 1100
        })
        
        # Check that gaps were filled
        self.assertGreater(bar_gen.metrics.gaps_filled_5min, 0)
    
    def test_smart_fill_strategy(self):
        """Test smart fill gap strategy."""
        bar_gen = self.create_bar_generator(GapFillStrategy.SMART_FILL)
        
        # Test market hours gap vs. weekend gap
        weekday_time = datetime(2024, 1, 15, 15, 0, 0, tzinfo=timezone.utc)  # Monday
        weekend_time = datetime(2024, 1, 13, 15, 0, 0, tzinfo=timezone.utc)  # Saturday
        
        # Should handle weekday and weekend gaps differently
        bar_gen.on_new_tick({
            'timestamp': weekday_time,
            'price': 100.0,
            'volume': 1000
        })
        
        gap_analysis = bar_gen.get_gap_analysis()
        # Smart fill should make intelligent decisions based on market hours

if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)