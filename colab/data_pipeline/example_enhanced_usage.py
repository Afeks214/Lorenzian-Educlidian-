#!/usr/bin/env python3
"""
Example Usage of Enhanced DataFlowCoordinator

This example demonstrates how to use the enhanced DataFlowCoordinator
with race condition fixes and dependency management.
"""

import sys
import os
import time
import threading
import tempfile
import shutil
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_flow_coordinator import (
    EnhancedDataFlowCoordinator,
    DataStreamType,
    DataStreamPriority,
    EnhancedCoordinatorConfig,
    create_enhanced_coordinator
)

def basic_usage_example():
    """Basic usage example of enhanced coordinator"""
    print("=== Basic Usage Example ===")
    
    # Create temporary directory for coordination
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create enhanced coordinator with custom configuration
        config = EnhancedCoordinatorConfig(
            coordination_dir=temp_dir,
            enable_persistence=True,
            max_concurrent_streams=50,
            deadlock_detection_interval=2.0,
            enable_performance_monitoring=True
        )
        
        coordinator = create_enhanced_coordinator(config)
        
        # Register notebooks
        coordinator.register_notebook(
            notebook_id="execution_engine",
            notebook_type="execution",
            capabilities=["data_processing", "model_training"]
        )
        
        coordinator.register_notebook(
            notebook_id="risk_management",
            notebook_type="risk",
            capabilities=["risk_calculation", "monitoring"]
        )
        
        # Create enhanced streams with dependencies
        print("\nCreating enhanced streams...")
        
        # Base data stream
        market_stream = coordinator.create_enhanced_stream(
            stream_id="market_data_stream",
            stream_type=DataStreamType.MARKET_DATA,
            producer_notebook="execution_engine",
            consumer_notebooks=["risk_management"],
            priority=DataStreamPriority.HIGH
        )
        
        # Feature stream that depends on market data
        feature_stream = coordinator.create_enhanced_stream(
            stream_id="feature_stream",
            stream_type=DataStreamType.FEATURES,
            producer_notebook="execution_engine",
            consumer_notebooks=["risk_management"],
            priority=DataStreamPriority.MEDIUM,
            dependencies=["market_data_stream"]
        )
        
        # Risk metrics stream that depends on features
        risk_stream = coordinator.create_enhanced_stream(
            stream_id="risk_metrics_stream",
            stream_type=DataStreamType.RISK_METRICS,
            producer_notebook="risk_management",
            consumer_notebooks=["execution_engine"],
            priority=DataStreamPriority.CRITICAL,
            dependencies=["feature_stream"]
        )
        
        print("✓ Enhanced streams created successfully")
        
        # Get execution order
        execution_order = coordinator.get_stream_execution_order()
        print(f"Optimal execution order: {execution_order}")
        
        # Check dependency graph
        dep_info = coordinator.get_dependency_graph_info()
        print(f"Dependency graph info: {dep_info}")
        
        # Publish data with dependencies
        print("\nPublishing data with dependency resolution...")
        
        # Publish market data
        success = coordinator.publish_with_dependencies(
            stream_id="market_data_stream",
            data={"symbol": "NQ", "price": 15000, "volume": 1000},
            priority=DataStreamPriority.HIGH
        )
        print(f"Market data published: {success}")
        
        # Publish features (depends on market data)
        success = coordinator.publish_with_dependencies(
            stream_id="feature_stream",
            data={"rsi": 55, "macd": 0.5, "volume_sma": 950},
            dependencies=["market_data_stream"],
            priority=DataStreamPriority.MEDIUM
        )
        print(f"Features published: {success}")
        
        # Publish risk metrics (depends on features)
        success = coordinator.publish_with_dependencies(
            stream_id="risk_metrics_stream",
            data={"var": 0.02, "sharpe": 1.5, "drawdown": 0.05},
            dependencies=["feature_stream"],
            priority=DataStreamPriority.CRITICAL
        )
        print(f"Risk metrics published: {success}")
        
        # Get coordination status
        status = coordinator.get_enhanced_coordination_status()
        print(f"\nCoordination status:")
        print(f"  Enhanced streams: {status['enhanced_streams']}")
        print(f"  Stream creations: {status['operation_counters']['stream_creations']}")
        print(f"  Message publishes: {status['operation_counters']['message_publishes']}")
        print(f"  Dependency resolutions: {status['operation_counters']['dependency_resolutions']}")
        
        # Demonstrate stream performance metrics
        print(f"\nStream performance metrics:")
        for stream_id, metrics in status['enhanced_stream_statistics'].items():
            print(f"  {stream_id}:")
            print(f"    Priority: {metrics['priority']}")
            print(f"    Messages sent: {metrics['messages_sent']}")
            print(f"    Avg processing time: {metrics['avg_processing_time']:.4f}s")
            print(f"    Throughput: {metrics['throughput_per_second']}")
        
        # Clean up
        coordinator.shutdown_enhanced()
        print("\n✓ Enhanced coordinator shutdown complete")
        
    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir)


def concurrent_usage_example():
    """Example of concurrent usage with race condition handling"""
    print("\n=== Concurrent Usage Example ===")
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create coordinator
        coordinator = EnhancedDataFlowCoordinator(
            coordination_dir=temp_dir,
            enable_persistence=True,
            max_concurrent_streams=100,
            deadlock_detection_interval=1.0,
            enable_performance_monitoring=True
        )
        
        # Register multiple notebooks
        notebooks = ["execution_1", "execution_2", "risk_1", "risk_2"]
        for notebook in notebooks:
            coordinator.register_notebook(
                notebook_id=notebook,
                notebook_type="execution" if "execution" in notebook else "risk",
                capabilities=["data_processing"]
            )
        
        print(f"Registered {len(notebooks)} notebooks")
        
        # Create streams concurrently
        def create_stream_worker(worker_id):
            try:
                stream_id = f"concurrent_stream_{worker_id}"
                coordinator.create_enhanced_stream(
                    stream_id=stream_id,
                    stream_type=DataStreamType.MARKET_DATA,
                    producer_notebook=f"execution_{worker_id % 2 + 1}",
                    consumer_notebooks=[f"risk_{worker_id % 2 + 1}"],
                    priority=DataStreamPriority.MEDIUM
                )
                print(f"✓ Created stream {stream_id}")
                
                # Publish messages
                for i in range(10):
                    coordinator.publish_with_dependencies(
                        stream_id=stream_id,
                        data=f"data_{worker_id}_{i}",
                        priority=DataStreamPriority.MEDIUM
                    )
                
            except Exception as e:
                print(f"✗ Error in worker {worker_id}: {e}")
        
        # Start concurrent workers
        threads = []
        for i in range(20):
            thread = threading.Thread(target=create_stream_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check final status
        status = coordinator.get_enhanced_coordination_status()
        print(f"\nConcurrent operation results:")
        print(f"  Total streams created: {status['enhanced_streams']}")
        print(f"  Total messages published: {status['operation_counters']['message_publishes']}")
        print(f"  Concurrency metrics: {status['concurrency_metrics']}")
        
        # Clean up
        coordinator.shutdown_enhanced()
        print("✓ Concurrent example completed successfully")
        
    finally:
        shutil.rmtree(temp_dir)


def dependency_management_example():
    """Example of advanced dependency management"""
    print("\n=== Dependency Management Example ===")
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        coordinator = EnhancedDataFlowCoordinator(
            coordination_dir=temp_dir,
            enable_persistence=True,
            max_concurrent_streams=50,
            deadlock_detection_interval=1.0,
            enable_performance_monitoring=True
        )
        
        # Register notebook
        coordinator.register_notebook("data_processor", "execution", ["processing"])
        
        # Create complex dependency chain
        print("Creating complex dependency chain...")
        
        streams = [
            ("raw_data", DataStreamType.MARKET_DATA, []),
            ("preprocessed", DataStreamType.FEATURES, ["raw_data"]),
            ("features", DataStreamType.FEATURES, ["preprocessed"]),
            ("predictions", DataStreamType.PREDICTIONS, ["features"]),
            ("risk_metrics", DataStreamType.RISK_METRICS, ["predictions", "features"])
        ]
        
        for stream_id, stream_type, dependencies in streams:
            try:
                coordinator.create_enhanced_stream(
                    stream_id=stream_id,
                    stream_type=stream_type,
                    producer_notebook="data_processor",
                    consumer_notebooks=["data_processor"],
                    dependencies=dependencies,
                    priority=DataStreamPriority.HIGH
                )
                print(f"✓ Created stream {stream_id} with dependencies: {dependencies}")
            except ValueError as e:
                print(f"✗ Failed to create stream {stream_id}: {e}")
        
        # Get optimal execution order
        execution_order = coordinator.get_stream_execution_order()
        print(f"\nOptimal execution order: {execution_order}")
        
        # Test circular dependency prevention
        print("\nTesting circular dependency prevention...")
        try:
            coordinator.create_enhanced_stream(
                stream_id="circular_test",
                stream_type=DataStreamType.MARKET_DATA,
                producer_notebook="data_processor",
                consumer_notebooks=["data_processor"],
                dependencies=["risk_metrics"]  # This would create a cycle
            )
            print("✗ Circular dependency not detected!")
        except ValueError as e:
            print(f"✓ Circular dependency correctly prevented: {e}")
        
        # Demonstrate dependency resolution
        print("\nTesting dependency resolution...")
        
        # Publish data in dependency order
        for stream_id in execution_order:
            success = coordinator.publish_with_dependencies(
                stream_id=stream_id,
                data=f"processed_data_for_{stream_id}",
                priority=DataStreamPriority.HIGH
            )
            print(f"Published to {stream_id}: {success}")
        
        # Get final dependency graph info
        dep_info = coordinator.get_dependency_graph_info()
        print(f"\nFinal dependency graph:")
        for stream_id, deps in dep_info['dependencies'].items():
            print(f"  {stream_id} depends on: {deps}")
        
        coordinator.shutdown_enhanced()
        print("✓ Dependency management example completed")
        
    finally:
        shutil.rmtree(temp_dir)


def main():
    """Main example runner"""
    print("Enhanced DataFlowCoordinator Examples")
    print("=" * 50)
    
    # Run examples
    basic_usage_example()
    concurrent_usage_example()
    dependency_management_example()
    
    print("\n" + "=" * 50)
    print("All examples completed successfully!")


if __name__ == "__main__":
    main()