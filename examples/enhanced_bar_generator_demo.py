#!/usr/bin/env python3
"""
Enhanced BarGenerator Demo
==========================

Demonstrates the enhanced BarGenerator with production-ready features:
- Timezone-aware timestamp handling
- Intelligent gap detection and filling
- Market hours awareness
- Data validation and integrity checks
- Performance monitoring
- Comprehensive reporting

Author: Agent 7 - Timestamp Alignment and Gap Handling Specialist
"""

import sys
import os
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from components.bar_generator import (
    BarGenerator, 
    BarGeneratorConfig, 
    GapFillStrategy,
    MarketSession
)

class MockEventBus:
    """Mock event bus for demonstration."""
    
    def __init__(self):
        self.events = []
    
    def publish(self, event_type: str, data: Any):
        self.events.append({
            'type': event_type,
            'data': data,
            'timestamp': datetime.now(timezone.utc)
        })
        print(f"Event: {event_type} - {data.timestamp} - OHLC: {data.open:.2f}/{data.high:.2f}/{data.low:.2f}/{data.close:.2f}")

def create_sample_tick_data() -> List[Dict[str, Any]]:
    """Create sample tick data with gaps for demonstration."""
    base_time = datetime(2024, 1, 15, 9, 30, 0, tzinfo=timezone.utc)
    ticks = []
    
    # Normal ticks
    for i in range(10):
        ticks.append({
            'timestamp': base_time + timedelta(minutes=i),
            'price': 100.0 + i * 0.1,
            'volume': 1000 + i * 10
        })
    
    # Gap of 30 minutes (data missing)
    gap_start = base_time + timedelta(minutes=10)
    gap_end = base_time + timedelta(minutes=40)
    
    # Resume after gap
    for i in range(10):
        ticks.append({
            'timestamp': gap_end + timedelta(minutes=i),
            'price': 101.0 + i * 0.1,
            'volume': 1100 + i * 10
        })
    
    # Weekend gap
    weekend_start = base_time + timedelta(days=2)  # Saturday
    weekend_end = base_time + timedelta(days=4)    # Monday
    
    for i in range(5):
        ticks.append({
            'timestamp': weekend_end + timedelta(minutes=i),
            'price': 102.0 + i * 0.1,
            'volume': 1200 + i * 10
        })
    
    return ticks

def demonstrate_enhanced_features():
    """Demonstrate enhanced BarGenerator features."""
    
    print("="*60)
    print("Enhanced BarGenerator Demonstration")
    print("="*60)
    
    # Create configuration
    config = BarGeneratorConfig(
        timezone="America/New_York",
        gap_fill_strategy=GapFillStrategy.SMART_FILL,
        max_gap_minutes=120,
        enable_market_hours=True,
        validate_timestamps=True,
        enable_data_quality_checks=True,
        performance_monitoring=True,
        duplicate_detection=True,
        max_out_of_order_seconds=10,
        synthetic_bar_volume_threshold=0.1
    )
    
    print(f"Configuration: {config.timezone}, {config.gap_fill_strategy.value}")
    print()
    
    # Create event bus and bar generator
    event_bus = MockEventBus()
    bar_generator = BarGenerator(config, event_bus)
    
    # Create sample data
    sample_ticks = create_sample_tick_data()
    
    print(f"Processing {len(sample_ticks)} sample ticks...")
    print()
    
    # Process ticks
    for tick in sample_ticks:
        bar_generator.on_new_tick(tick)
    
    print()
    print("="*60)
    print("Statistics and Reports")
    print("="*60)
    
    # Get statistics
    stats = bar_generator.get_statistics()
    print("Basic Statistics:")
    print(f"  - Ticks processed: {stats.get('tick_count', 0)}")
    print(f"  - 5-min bars emitted: {stats.get('bars_emitted_5min', 0)}")
    print(f"  - 30-min bars emitted: {stats.get('bars_emitted_30min', 0)}")
    print(f"  - Gaps filled (5-min): {stats.get('gaps_filled_5min', 0)}")
    print(f"  - Gaps filled (30-min): {stats.get('gaps_filled_30min', 0)}")
    print(f"  - Validation errors: {stats.get('validation_errors', 0)}")
    print(f"  - Duplicate ticks: {stats.get('duplicate_ticks', 0)}")
    print()
    
    # Get gap analysis
    gap_analysis = bar_generator.get_gap_analysis()
    print("Gap Analysis:")
    print(f"  - Total gaps detected: {gap_analysis['gap_statistics']['total_gaps']}")
    print(f"  - Gap types: {gap_analysis['gap_statistics']['gap_types']}")
    print(f"  - Fill strategies: {gap_analysis['gap_statistics']['fill_strategies']}")
    print()
    
    # Get data quality report
    quality_report = bar_generator.get_data_quality_report()
    print("Data Quality Report:")
    print(f"  - Data completeness: {quality_report['overall_quality']['data_completeness']:.2%}")
    print(f"  - Timestamp accuracy: {quality_report['overall_quality']['timestamp_accuracy']:.2%}")
    print(f"  - Data integrity: {quality_report['overall_quality']['data_integrity']:.2%}")
    print(f"  - Duplicate cleanliness: {quality_report['overall_quality']['duplicate_cleanliness']:.2%}")
    print()
    
    print("Quality Recommendations:")
    for recommendation in quality_report['recommendations']:
        print(f"  - {recommendation}")
    print()
    
    # Get system health
    health_report = bar_generator.get_system_health()
    print("System Health:")
    print(f"  - Overall health score: {health_report['overall_health_score']:.2%}")
    print(f"  - System status: {health_report['system_status']}")
    print(f"  - Processed ticks: {health_report['processed_ticks']}")
    print(f"  - Memory usage: {health_report['memory_usage_mb']:.1f} MB")
    print()
    
    if health_report['alerts']:
        print("Health Alerts:")
        for alert in health_report['alerts']:
            print(f"  ⚠️  {alert}")
    else:
        print("✅ No health alerts")
    
    print()
    
    # Configuration summary
    config_summary = bar_generator.get_configuration_summary()
    print("Configuration Summary:")
    for key, value in config_summary.items():
        print(f"  - {key}: {value}")
    
    print()
    print("="*60)
    print("Event Bus Activity")
    print("="*60)
    
    print(f"Total events published: {len(event_bus.events)}")
    
    # Show sample events
    event_types = {}
    for event in event_bus.events:
        event_type = event['type']
        event_types[event_type] = event_types.get(event_type, 0) + 1
    
    print("Event types:")
    for event_type, count in event_types.items():
        print(f"  - {event_type}: {count}")
    
    print()
    print("Last 5 events:")
    for event in event_bus.events[-5:]:
        data = event['data']
        synthetic_label = " (synthetic)" if data.is_synthetic else ""
        print(f"  - {event['type']}: {data.timestamp} - {data.market_session.value}{synthetic_label}")
    
    print()
    print("="*60)
    print("Enhanced BarGenerator Demo Complete")
    print("="*60)

def demonstrate_timezone_handling():
    """Demonstrate timezone-aware timestamp handling."""
    print("\n" + "="*60)
    print("Timezone Handling Demonstration")
    print("="*60)
    
    # Test different timezone configurations
    timezones = ["America/New_York", "Europe/London", "Asia/Tokyo", "UTC"]
    
    for tz in timezones:
        print(f"\nTesting timezone: {tz}")
        
        config = BarGeneratorConfig(timezone=tz)
        event_bus = MockEventBus()
        bar_generator = BarGenerator(config, event_bus)
        
        # Create tick with naive timestamp
        naive_time = datetime(2024, 1, 15, 14, 30, 0)  # 2:30 PM, no timezone
        tick = {
            'timestamp': naive_time,
            'price': 100.0,
            'volume': 1000
        }
        
        print(f"  Input timestamp (naive): {naive_time}")
        bar_generator.on_new_tick(tick)
        
        if event_bus.events:
            bar_data = event_bus.events[-1]['data']
            print(f"  Processed timestamp: {bar_data.timestamp}")
            print(f"  Timezone: {bar_data.timestamp.tzinfo}")
            print(f"  Market session: {bar_data.market_session.value}")
        
        event_bus.events.clear()

def demonstrate_gap_strategies():
    """Demonstrate different gap filling strategies."""
    print("\n" + "="*60)
    print("Gap Filling Strategies Demonstration")
    print("="*60)
    
    strategies = [
        GapFillStrategy.FORWARD_FILL,
        GapFillStrategy.ZERO_VOLUME,
        GapFillStrategy.SMART_FILL,
        GapFillStrategy.SKIP
    ]
    
    for strategy in strategies:
        print(f"\nTesting strategy: {strategy.value}")
        
        config = BarGeneratorConfig(gap_fill_strategy=strategy)
        event_bus = MockEventBus()
        bar_generator = BarGenerator(config, event_bus)
        
        # Create ticks with gap
        base_time = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        
        # First tick
        bar_generator.on_new_tick({
            'timestamp': base_time,
            'price': 100.0,
            'volume': 1000
        })
        
        # Second tick after 15-minute gap
        bar_generator.on_new_tick({
            'timestamp': base_time + timedelta(minutes=15),
            'price': 101.0,
            'volume': 1100
        })
        
        gap_analysis = bar_generator.get_gap_analysis()
        print(f"  Gaps detected: {gap_analysis['gap_statistics']['total_gaps']}")
        
        stats = bar_generator.get_statistics()
        print(f"  Bars emitted: {stats.get('bars_emitted_5min', 0)}")
        print(f"  Gaps filled: {stats.get('gaps_filled_5min', 0)}")

if __name__ == "__main__":
    try:
        demonstrate_enhanced_features()
        demonstrate_timezone_handling()
        demonstrate_gap_strategies()
        
        print("\n🎉 All demonstrations completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()