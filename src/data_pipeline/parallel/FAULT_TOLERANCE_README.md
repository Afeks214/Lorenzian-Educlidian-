# Fault Tolerance and Health Monitoring Implementation

## Overview

This implementation adds comprehensive fault tolerance and health monitoring capabilities to the ParallelProcessor component, ensuring high availability and reliability in production environments.

## Features

### 1. Worker Failure Detection
- **Heartbeat Monitoring**: Continuous monitoring of worker heartbeats
- **Timeout Detection**: Automatic detection of unresponsive workers
- **Health State Tracking**: Real-time worker state management
- **Resource Monitoring**: CPU, memory, and disk usage tracking

### 2. Automatic Worker Recovery
- **Failed Worker Detection**: Immediate detection of worker failures
- **Task Reassignment**: Automatic reassignment of tasks from failed workers
- **Worker Replacement**: Spawning of replacement workers when needed
- **Graceful Degradation**: Continued operation with reduced worker count

### 3. Task Retry Mechanisms
- **Exponential Backoff**: Intelligent retry delays with exponential backoff
- **Retry Limits**: Configurable maximum retry attempts
- **Jitter Prevention**: Anti-thundering herd protection
- **State Persistence**: Task state tracking throughout retry cycles

### 4. Health Monitoring System
- **Real-time Metrics**: Continuous collection of worker health metrics
- **Performance Tracking**: Monitoring of task execution times and throughput
- **Degradation Detection**: Early warning system for performance issues
- **Resource Utilization**: Comprehensive system resource monitoring

### 5. Checkpoint/Restart Mechanism
- **Automatic Checkpointing**: Periodic saving of processing state
- **Crash Recovery**: Automatic restoration from checkpoints
- **State Persistence**: Preservation of task metadata and progress
- **Configurable Intervals**: Adjustable checkpoint frequency

### 6. Alerting System
- **Multi-level Alerts**: Info, Warning, Error, and Critical alerts
- **Component-specific**: Alerts tagged by component and worker
- **Threshold-based**: Configurable alert thresholds
- **Historical Tracking**: Alert history and resolution tracking

## Configuration

### Fault Tolerance Settings

```python
config = ParallelConfig(
    # Basic settings
    max_workers=8,
    enable_fault_tolerance=True,
    enable_health_monitoring=True,
    enable_checkpointing=True,
    enable_alerting=True,
    enable_auto_recovery=True,
    
    # Retry configuration
    max_task_retries=3,
    retry_delay_seconds=1.0,
    retry_backoff_factor=2.0,
    
    # Health monitoring
    worker_timeout_seconds=60.0,
    heartbeat_interval_seconds=5.0,
    health_check_interval_seconds=10.0,
    
    # Thresholds
    cpu_threshold_percent=80.0,
    memory_threshold_percent=75.0,
    disk_threshold_percent=85.0,
    error_rate_threshold=0.05,
    
    # Checkpointing
    checkpoint_interval_seconds=30.0,
    checkpoint_directory="/tmp/parallel_processor_checkpoints",
    
    # Alert thresholds
    alert_thresholds={
        'cpu_critical': 90.0,
        'memory_critical': 85.0,
        'error_rate_critical': 0.1,
        'worker_timeout_critical': 120.0
    }
)
```

## Worker States

The system tracks workers through the following states:

- **INITIALIZING**: Worker is starting up
- **HEALTHY**: Worker is operating normally
- **DEGRADED**: Worker performance is below optimal
- **UNHEALTHY**: Worker has significant issues
- **FAILED**: Worker has failed completely
- **RECOVERING**: Worker is being recovered
- **TERMINATED**: Worker has been shut down

## Task States

Tasks progress through these states:

- **PENDING**: Task is waiting to be assigned
- **ASSIGNED**: Task has been assigned to a worker
- **RUNNING**: Task is currently being processed
- **COMPLETED**: Task has been successfully completed
- **FAILED**: Task has failed processing
- **RETRYING**: Task is being retried after failure
- **CANCELLED**: Task has been cancelled

## Alert Levels

The system generates alerts at different severity levels:

- **INFO**: Informational messages
- **WARNING**: Performance degradation or minor issues
- **ERROR**: Serious problems requiring attention
- **CRITICAL**: System-threatening issues requiring immediate action

## Usage Examples

### Basic Usage with Fault Tolerance

```python
from parallel_processor import ParallelProcessor, ParallelConfig

# Create fault-tolerant configuration
config = ParallelConfig(
    max_workers=4,
    enable_fault_tolerance=True,
    enable_health_monitoring=True,
    max_task_retries=3
)

# Create processor
processor = ParallelProcessor(config)

# Define processing function
def process_chunk(chunk):
    # Your processing logic here
    return processed_chunk

# Process with fault tolerance
file_paths = ["file1.txt", "file2.txt", "file3.txt"]
results = list(processor.process_files_parallel(file_paths, process_chunk))

# Check health status
health_status = processor.get_health_status()
print(f"Healthy workers: {health_status['healthy_workers']}")

# Get comprehensive statistics
stats = processor.get_stats()
print(f"Worker health: {stats['worker_health']}")
print(f"Fault tolerance: {stats['fault_tolerance']}")
print(f"Alerts: {stats['alerts']}")

# Shutdown gracefully
processor.shutdown()
```

### Monitoring and Alerting

```python
# Check active alerts
if processor.config.enable_alerting:
    active_alerts = processor.alert_manager.get_active_alerts()
    for alert in active_alerts:
        print(f"[{alert.level.value}] {alert.message}")

# Monitor worker health
for worker_id, metrics in processor.worker_metrics.items():
    print(f"Worker {worker_id}: {metrics.state.value}")
    print(f"  CPU: {metrics.cpu_percent}%")
    print(f"  Memory: {metrics.memory_percent}%")
    print(f"  Error rate: {metrics.error_rate}")
```

### Checkpoint Recovery

```python
# The processor automatically loads checkpoints on startup
processor = ParallelProcessor(config)

# Checkpoints are created automatically during processing
# Manual checkpoint creation is also possible
processor.checkpoint_manager.create_checkpoint(processor.active_tasks)
```

## Monitoring Integration

### Metrics Collection

The system provides comprehensive metrics for monitoring:

```python
stats = processor.get_stats()

# Base metrics
print(f"Chunks processed: {stats['chunks_processed']}")
print(f"Throughput: {stats['throughput']} chunks/second")

# Worker health metrics
if 'worker_health' in stats:
    health = stats['worker_health']
    print(f"Total workers: {health['total_workers']}")
    print(f"Healthy workers: {health['healthy_workers']}")
    print(f"Average CPU: {health['avg_cpu_percent']}%")
    print(f"Average memory: {health['avg_memory_percent']}%")

# Fault tolerance metrics
if 'fault_tolerance' in stats:
    ft = stats['fault_tolerance']
    print(f"Active tasks: {ft['active_tasks']}")
    print(f"Failed tasks: {ft['failed_tasks']}")
    print(f"Retrying tasks: {ft['retrying_tasks']}")
```

### Health Status API

```python
health_status = processor.get_health_status()
print(f"System running: {health_status['is_running']}")
print(f"Worker count: {health_status['worker_count']}")
print(f"Healthy workers: {health_status['healthy_workers']}")
print(f"Active tasks: {health_status['active_tasks']}")
print(f"Active alerts: {health_status['active_alerts']}")
```

## Architecture Components

### Core Classes

1. **ParallelProcessor**: Main processor with fault tolerance integration
2. **WorkerHealthMonitor**: Monitors worker health and performance
3. **TaskManager**: Manages task lifecycle and recovery
4. **AlertManager**: Handles alert generation and management
5. **CheckpointManager**: Manages checkpointing and recovery

### Data Structures

1. **WorkerHealthMetrics**: Comprehensive worker health data
2. **TaskMetadata**: Task state and recovery information
3. **Alert**: Alert information and metadata
4. **WorkerState**: Worker state enumeration
5. **TaskState**: Task state enumeration
6. **AlertLevel**: Alert severity levels

## Best Practices

### Configuration

1. **Tune Retry Settings**: Adjust retry count and delays based on workload
2. **Monitor Resource Usage**: Set appropriate thresholds for your environment
3. **Configure Checkpointing**: Balance checkpoint frequency with performance
4. **Alert Thresholds**: Set meaningful thresholds for your use case

### Error Handling

1. **Graceful Degradation**: Design processing functions to handle partial failures
2. **Idempotent Operations**: Ensure processing functions can be safely retried
3. **Resource Cleanup**: Properly clean up resources in processing functions
4. **Logging**: Use appropriate logging levels for debugging

### Performance

1. **Resource Monitoring**: Monitor CPU, memory, and disk usage
2. **Batch Size Optimization**: Adjust batch sizes based on resource availability
3. **Worker Count**: Optimize worker count based on workload characteristics
4. **Checkpoint Frequency**: Balance durability with performance impact

### Production Deployment

1. **External Monitoring**: Integrate with external monitoring systems
2. **Log Aggregation**: Collect and analyze logs centrally
3. **Alert Routing**: Route alerts to appropriate incident management systems
4. **Capacity Planning**: Monitor trends and plan for capacity needs

## Troubleshooting

### Common Issues

1. **High Memory Usage**: Check for memory leaks in processing functions
2. **Worker Failures**: Monitor system resources and error logs
3. **Task Retries**: Investigate root causes of task failures
4. **Checkpoint Issues**: Verify checkpoint directory permissions and disk space

### Debugging

1. **Enable Debug Logging**: Set logging level to DEBUG for detailed information
2. **Monitor Metrics**: Use provided metrics to identify bottlenecks
3. **Check Alerts**: Review alert history for patterns
4. **Analyze Checkpoints**: Examine checkpoint files for task state information

## Integration Examples

### With Monitoring Systems

```python
# Prometheus metrics integration example
def export_metrics_to_prometheus(processor):
    stats = processor.get_stats()
    
    # Export worker health metrics
    if 'worker_health' in stats:
        health = stats['worker_health']
        prometheus_client.Gauge('parallel_processor_healthy_workers').set(health['healthy_workers'])
        prometheus_client.Gauge('parallel_processor_cpu_percent').set(health['avg_cpu_percent'])
        prometheus_client.Gauge('parallel_processor_memory_percent').set(health['avg_memory_percent'])
    
    # Export fault tolerance metrics
    if 'fault_tolerance' in stats:
        ft = stats['fault_tolerance']
        prometheus_client.Gauge('parallel_processor_active_tasks').set(ft['active_tasks'])
        prometheus_client.Gauge('parallel_processor_failed_tasks').set(ft['failed_tasks'])
```

### With Alerting Systems

```python
# PagerDuty integration example
def send_alerts_to_pagerduty(processor):
    if processor.config.enable_alerting:
        active_alerts = processor.alert_manager.get_active_alerts()
        
        for alert in active_alerts:
            if alert.level == AlertLevel.CRITICAL:
                # Send to PagerDuty
                send_pagerduty_alert(alert.message, alert.metadata)
```

## Future Enhancements

### Planned Features

1. **Circuit Breaker Pattern**: Prevent cascading failures
2. **Load Balancing**: Intelligent task distribution
3. **Auto-scaling**: Dynamic worker count adjustment
4. **Metrics Export**: Native Prometheus/Grafana integration
5. **Advanced Recovery**: Sophisticated recovery strategies

### Extension Points

1. **Custom Health Checks**: Pluggable health check implementations
2. **Alert Handlers**: Custom alert processing and routing
3. **Checkpoint Formats**: Alternative checkpoint storage formats
4. **Recovery Strategies**: Customizable recovery algorithms

This implementation provides a robust foundation for fault-tolerant parallel processing in production environments, with comprehensive monitoring, alerting, and recovery capabilities.