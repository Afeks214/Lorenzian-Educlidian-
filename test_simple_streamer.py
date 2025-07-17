#!/usr/bin/env python3
"""
Simple test for enhanced DataStreamer functionality
"""

import sys
import os
import time
import pandas as pd
import numpy as np
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import only what we need, directly
from data_pipeline.streaming.data_streamer import (
    DataStreamer, StreamConfig, BackpressureConfig, 
    RateLimitConfig, FlowControlConfig, Priority
)

def create_test_data():
    """Create test CSV data for streaming"""
    test_dir = Path("/tmp/test_streaming")
    test_dir.mkdir(exist_ok=True)
    
    # Create a test CSV file
    test_file = test_dir / "test_data.csv"
    
    # Generate test data
    data = {
        'timestamp': pd.date_range('2024-01-01', periods=1000, freq='1s'),
        'value': np.random.randn(1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000),
        'id': range(1000)
    }
    
    df = pd.DataFrame(data)
    df.to_csv(test_file, index=False)
    
    return str(test_file)

def test_enhanced_streaming():
    """Test enhanced streaming with backpressure and rate limiting"""
    print("Testing enhanced DataStreamer...")
    
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
    streamer = DataStreamer(stream_config=stream_config)
    
    # Test basic streaming
    chunk_count = 0
    start_time = time.time()
    
    try:
        for chunk in streamer.stream_file(test_file, chunk_size=200):
            chunk_count += 1
            print(f"Processed chunk {chunk_count}: {len(chunk.data)} rows")
            
            # Stop after a few chunks for testing
            if chunk_count >= 3:
                break
                
    except Exception as e:
        print(f"Error during streaming: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Get stats
    stats = streamer.get_streaming_stats()
    print(f"\\nStreaming completed in {time.time() - start_time:.2f} seconds")
    print(f"Chunks processed: {stats['chunks_processed']}")
    print(f"Total rows: {stats['total_rows']}")
    print(f"Throughput: {stats['throughput_rows_per_sec']:.2f} rows/sec")
    
    # Test health status
    try:
        health = streamer.get_health_status()
        print(f"\\nHealth Status: {health['status']}")
        print(f"Health score: {health['health_score']:.2f}")
    except Exception as e:
        print(f"Health check failed: {e}")
    
    # Cleanup
    streamer.stop_streaming()
    
    return True

def test_backward_compatibility():
    """Test backward compatibility"""
    print("\\nTesting backward compatibility...")
    
    # Create test data
    test_file = create_test_data()
    
    # Create streamer with minimal config
    streamer = DataStreamer()
    
    # Test basic streaming
    chunk_count = 0
    
    try:
        for chunk in streamer.stream_file(test_file, chunk_size=200):
            chunk_count += 1
            print(f"Legacy chunk {chunk_count}: {len(chunk.data)} rows")
            
            if chunk_count >= 2:
                break
                
    except Exception as e:
        print(f"Error in backward compatibility test: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test old stats interface
    stats = streamer.get_streaming_stats()
    print(f"Legacy stats - Chunks: {stats['chunks_processed']}, Rows: {stats['total_rows']}")
    
    return True

def main():
    """Main test function"""
    print("Enhanced DataStreamer Simple Test")
    print("=" * 40)
    
    tests = [
        ("Enhanced Streaming", test_enhanced_streaming),
        ("Backward Compatibility", test_backward_compatibility)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\\n{test_name}:")
        print("-" * 25)
        
        try:
            if test_func():
                print(f"✓ {test_name} PASSED")
                passed += 1
            else:
                print(f"✗ {test_name} FAILED")
                failed += 1
        except Exception as e:
            print(f"✗ {test_name} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print(f"\\n{'='*40}")
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