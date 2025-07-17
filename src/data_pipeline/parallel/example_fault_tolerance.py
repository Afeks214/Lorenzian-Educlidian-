"""
Example demonstrating fault tolerance and health monitoring features in ParallelProcessor
"""

import time
import logging
import random
from pathlib import Path
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

from parallel_processor import (
    ParallelProcessor,
    ParallelConfig,
    DataChunk,
    WorkerState,
    TaskState,
    AlertLevel
)

def create_sample_processing_function():
    """Create a processing function that sometimes fails"""
    def process_chunk(chunk: DataChunk) -> DataChunk:
        # Simulate processing time
        time.sleep(random.uniform(0.1, 0.5))
        
        # Simulate random failures (10% chance)
        if random.random() < 0.1:
            raise Exception(f"Simulated failure processing chunk {chunk.chunk_id}")
        
        # Simulate high memory usage occasionally
        if random.random() < 0.05:
            # Simulate memory-intensive operation
            large_data = [i for i in range(100000)]
            del large_data
        
        # Return processed chunk
        return DataChunk(
            chunk_id=chunk.chunk_id,
            data=f"processed_{chunk.data}",
            metadata={'processing_time': time.time()}
        )
    
    return process_chunk

def create_sample_chunks(num_chunks: int = 20) -> List[DataChunk]:
    """Create sample data chunks for processing"""
    chunks = []
    for i in range(num_chunks):
        chunk = DataChunk(
            chunk_id=f"chunk_{i}",
            data=f"sample_data_{i}",
            metadata={'created_at': time.time()}
        )
        chunks.append(chunk)
    return chunks

def demonstrate_fault_tolerance():
    """Demonstrate fault tolerance features"""
    print("=== Fault Tolerance and Health Monitoring Demo ===\n")
    
    # Create configuration with fault tolerance enabled
    config = ParallelConfig(
        max_workers=4,
        use_processes=True,
        enable_fault_tolerance=True,
        enable_health_monitoring=True,
        enable_checkpointing=True,
        enable_alerting=True,
        enable_auto_recovery=True,
        
        # Fault tolerance settings
        max_task_retries=3,
        retry_delay_seconds=0.5,
        retry_backoff_factor=2.0,
        worker_timeout_seconds=30.0,
        heartbeat_interval_seconds=2.0,
        health_check_interval_seconds=5.0,
        
        # Health monitoring thresholds
        cpu_threshold_percent=70.0,
        memory_threshold_percent=60.0,
        error_rate_threshold=0.05,
        
        # Checkpoint settings
        checkpoint_interval_seconds=10.0,
        checkpoint_directory="/tmp/parallel_processor_demo_checkpoints"
    )
    
    # Create processor
    processor = ParallelProcessor(config)
    
    # Create sample processing function
    processing_func = create_sample_processing_function()
    
    # Create sample data chunks
    chunks = create_sample_chunks(50)
    
    # Simulate file paths (in real usage, these would be actual file paths)
    file_paths = ["sample_file_1.txt", "sample_file_2.txt", "sample_file_3.txt"]
    
    print("Starting parallel processing with fault tolerance...")
    start_time = time.time()
    
    try:
        # Process chunks with fault tolerance
        processed_chunks = []
        for chunk in processor.process_files_parallel(file_paths, processing_func):
            processed_chunks.append(chunk)
            
            # Print progress
            if len(processed_chunks) % 10 == 0:
                print(f"Processed {len(processed_chunks)} chunks...")
                
                # Show health status
                health_status = processor.get_health_status()
                print(f"Health Status: {health_status}")
                
                # Show statistics
                stats = processor.get_stats()
                print(f"Worker Health: {stats.get('worker_health', {})}")
                print(f"Fault Tolerance: {stats.get('fault_tolerance', {})}")
                print(f"Alerts: {stats.get('alerts', {})}")
                print()
        
        end_time = time.time()
        print(f"\nProcessing completed in {end_time - start_time:.2f} seconds")
        print(f"Successfully processed {len(processed_chunks)} chunks")
        
        # Show final statistics
        final_stats = processor.get_stats()
        print("\n=== Final Statistics ===")
        print(f"Total chunks processed: {final_stats['chunks_processed']}")
        print(f"Average processing time: {final_stats['avg_processing_time']:.4f}s")
        print(f"Throughput: {final_stats['throughput']:.2f} chunks/second")
        
        if 'worker_health' in final_stats:
            health = final_stats['worker_health']
            print(f"\nWorker Health:")
            print(f"  Total workers: {health['total_workers']}")
            print(f"  Healthy workers: {health['healthy_workers']}")
            print(f"  Degraded workers: {health['degraded_workers']}")
            print(f"  Unhealthy workers: {health['unhealthy_workers']}")
            print(f"  Average CPU usage: {health['avg_cpu_percent']:.1f}%")
            print(f"  Average memory usage: {health['avg_memory_percent']:.1f}%")
            print(f"  Average error rate: {health['avg_error_rate']:.3f}")
        
        if 'fault_tolerance' in final_stats:
            ft = final_stats['fault_tolerance']
            print(f"\nFault Tolerance:")
            print(f"  Active tasks: {ft['active_tasks']}")
            print(f"  Failed tasks: {ft['failed_tasks']}")
            print(f"  Retrying tasks: {ft['retrying_tasks']}")
        
        if 'alerts' in final_stats:
            alerts = final_stats['alerts']
            print(f"\nAlerts:")
            print(f"  Total active alerts: {alerts['total_active_alerts']}")
            print(f"  Critical alerts: {alerts['critical_alerts']}")
            print(f"  Error alerts: {alerts['error_alerts']}")
            print(f"  Warning alerts: {alerts['warning_alerts']}")
        
        # Show active alerts
        if processor.config.enable_alerting:
            active_alerts = processor.alert_manager.get_active_alerts()
            if active_alerts:
                print(f"\n=== Active Alerts ===")
                for alert in active_alerts[-5:]:  # Show last 5 alerts
                    print(f"[{alert.level.value.upper()}] {alert.message} (Worker: {alert.worker_id or 'N/A'})")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        
        # Show error statistics
        error_stats = processor.get_stats()
        print(f"Error statistics: {error_stats}")
    
    finally:
        # Shutdown processor
        processor.shutdown()
        print("\nProcessor shutdown completed")

def demonstrate_recovery_from_checkpoint():
    """Demonstrate recovery from checkpoint"""
    print("\n=== Checkpoint Recovery Demo ===\n")
    
    # Create configuration with checkpointing enabled
    config = ParallelConfig(
        max_workers=2,
        enable_checkpointing=True,
        checkpoint_interval_seconds=5.0,
        checkpoint_directory="/tmp/parallel_processor_demo_checkpoints"
    )
    
    # Create processor
    processor = ParallelProcessor(config)
    
    # Check if checkpoints exist
    checkpoint_dir = Path(config.checkpoint_directory)
    if checkpoint_dir.exists():
        checkpoint_files = list(checkpoint_dir.glob("checkpoint_*.json"))
        if checkpoint_files:
            print(f"Found {len(checkpoint_files)} checkpoint files")
            print("Latest checkpoint will be loaded automatically")
        else:
            print("No checkpoint files found")
    
    # The processor will automatically load checkpoints during initialization
    print("Processor initialized with checkpoint recovery")
    
    # Clean up
    processor.shutdown()

if __name__ == "__main__":
    # Run demonstrations
    demonstrate_fault_tolerance()
    demonstrate_recovery_from_checkpoint()
    
    print("\n=== Demo Complete ===")