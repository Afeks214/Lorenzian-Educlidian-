#!/usr/bin/env python3
"""
Test script for enhanced DataStreamer with backpressure and rate limiting
"""

import sys
import os
import time
import pandas as pd
import numpy as np
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_pipeline.streaming.data_streamer import (
    DataStreamer, StreamConfig, BackpressureConfig, 
    RateLimitConfig, FlowControlConfig, Priority
)
from data_pipeline.core.config import DataPipelineConfig

def create_test_data():
    """Create test CSV data for streaming"""
    test_dir = Path("/tmp/test_streaming")
    test_dir.mkdir(exist_ok=True)
    
    # Create a test CSV file
    test_file = test_dir / "test_data.csv"
    
    # Generate test data
    data = {
        'timestamp': pd.date_range('2024-01-01', periods=10000, freq='1s'),
        'value': np.random.randn(10000),
        'category': np.random.choice(['A', 'B', 'C'], 10000),
        'id': range(10000)
    }
    
    df = pd.DataFrame(data)
    df.to_csv(test_file, index=False)
    
    return str(test_file)

def test_basic_functionality():
    """Test basic enhanced streaming functionality"""
    print("Testing basic enhanced streaming functionality...")
    
    # Create test data
    test_file = create_test_data()
    
    # Create enhanced configuration
    backpressure_config = BackpressureConfig(
        enable_backpressure=True,
        max_queue_size=50,
        adaptive_threshold=True
    )
    
    rate_limit_config = RateLimitConfig(
        enable_rate_limiting=True,
        bucket_capacity=100,
        refill_rate=50.0,
        adaptive_rate_limiting=True
    )
    
    flow_control_config = FlowControlConfig(
        enable_flow_control=True,
        congestion_threshold=0.8,
        load_shed_threshold=0.9
    )
    
    stream_config = StreamConfig(
        buffer_size=500,
        max_queue_size=20,
        backpressure_config=backpressure_config,
        rate_limit_config=rate_limit_config,
        flow_control_config=flow_control_config
    )
    
    # Create streamer
    streamer = DataStreamer(
        config=DataPipelineConfig.for_development(),
        stream_config=stream_config
    )
    
    # Test streaming with priority
    chunk_count = 0
    start_time = time.time()
    
    try:
        for chunk in streamer.stream_with_priority(
            test_file, 
            priority=Priority.HIGH,
            chunk_size=1000
        ):
            chunk_count += 1
            print(f"Processed chunk {chunk_count}: {len(chunk.data)} rows")
            
            # Stop after a few chunks for testing
            if chunk_count >= 3:
                break
                
    except Exception as e:
        print(f"Error during streaming: {e}")
        return False
    
    # Get comprehensive stats
    stats = streamer.get_streaming_stats()
    print(f"\nStreaming completed in {time.time() - start_time:.2f} seconds")
    print(f"Chunks processed: {stats['chunks_processed']}")
    print(f"Total rows: {stats['total_rows']}")
    print(f"Throughput: {stats['throughput_rows_per_sec']:.2f} rows/sec")
    
    # Test performance summary
    perf_summary = streamer.get_performance_summary()
    print(f"\nPerformance Summary:")
    print(f"- Throughput: {perf_summary['throughput']:.2f} rows/sec")
    print(f"- Memory utilization: {perf_summary['memory_utilization']:.1f}%")
    print(f"- Backpressure active: {perf_summary['backpressure_active']}")
    print(f"- Rate limiting active: {perf_summary['rate_limiting_active']}")
    
    # Test health status
    health = streamer.get_health_status()
    print(f"\nHealth Status:")
    print(f"- Status: {health['status']}")
    print(f"- Health score: {health['health_score']:.2f}")
    print(f"- Issues: {health['issues']}")
    
    # Cleanup
    streamer.stop_streaming()
    
    return True

def test_backward_compatibility():
    """Test backward compatibility with existing code"""
    print("\nTesting backward compatibility...")
    
    # Create test data
    test_file = create_test_data()
    
    # Create streamer with minimal config (should work like before)
    streamer = DataStreamer()
    
    # Test basic streaming (old interface)
    chunk_count = 0
    
    try:
        for chunk in streamer.stream_file(test_file, chunk_size=1000):
            chunk_count += 1
            print(f"Legacy chunk {chunk_count}: {len(chunk.data)} rows")
            
            if chunk_count >= 2:
                break
                
    except Exception as e:
        print(f"Error in backward compatibility test: {e}")
        return False
    
    # Test old stats interface
    stats = streamer.get_streaming_stats()
    print(f"Legacy stats - Chunks: {stats['chunks_processed']}, Rows: {stats['total_rows']}")
    
    return True

def test_configuration_updates():
    """Test dynamic configuration updates"""
    print("\nTesting dynamic configuration updates...")
    
    streamer = DataStreamer()
    
    # Test dynamic backpressure configuration
    streamer.configure_backpressure(
        queue_size_threshold=0.9,
        adaptive_threshold=False
    )
    
    # Test dynamic rate limiting configuration
    streamer.configure_rate_limiting(
        bucket_capacity=200,
        refill_rate=75.0
    )
    
    # Test dynamic flow control configuration
    streamer.configure_flow_control(
        congestion_threshold=0.7,
        load_shed_threshold=0.85
    )
    
    print("Configuration updates completed successfully")
    return True

def main():
    """Main test function"""
    print("Enhanced DataStreamer Test Suite")
    print("=" * 50)
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Backward Compatibility", test_backward_compatibility),
        ("Configuration Updates", test_configuration_updates)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 30)
        
        try:
            if test_func():
                print(f"✓ {test_name} PASSED")
                passed += 1
            else:
                print(f"✗ {test_name} FAILED")
                failed += 1
        except Exception as e:
            print(f"✗ {test_name} FAILED with exception: {e}")
            failed += 1
    
    print(f"\n{'='*50}")
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("All tests passed! Enhanced DataStreamer is working correctly.")
        return True
    else:
        print("Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)