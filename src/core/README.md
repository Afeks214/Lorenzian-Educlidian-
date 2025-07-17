# Core Components

## Overview

The core components module contains the fundamental building blocks of the GrandModel trading system. These components provide the essential infrastructure for system orchestration, event management, and configuration handling.

## Components

### AlgoSpace Kernel (`kernel.py`)

The central orchestrator that manages the entire system lifecycle.

**Key Features:**
- Component lifecycle management
- Dependency resolution and initialization ordering
- Event bus coordination
- Configuration management
- Graceful shutdown handling

**Usage:**
```python
from src.core.kernel import AlgoSpaceKernel

# Initialize with configuration
kernel = AlgoSpaceKernel("configs/production.yaml")

# Initialize all components
kernel.initialize()

# Start the system
kernel.run()
```

**Configuration:**
```yaml
system:
  environment: production
  log_level: INFO

data_handler:
  type: rithmic
  symbols: ["ES", "NQ"]

strategic_marl:
  enabled: true
  model_path: "models/strategic_agent.pth"
```

### Event System (`events.py`)

High-performance event-driven communication system.

**Key Features:**
- Type-safe event definitions
- Asynchronous event processing
- Event performance monitoring
- Error isolation between subscribers

**Event Types:**
- `NEW_TICK`: Real-time market data updates
- `NEW_5MIN_BAR` / `NEW_30MIN_BAR`: Time-based bar completion
- `INDICATORS_READY`: Technical analysis completion
- `SYNERGY_DETECTED`: Pattern recognition alerts
- `EXECUTE_TRADE`: Trading decision execution
- `RISK_BREACH`: Risk management alerts

**Usage:**
```python
from src.core.events import EventBus, EventType, Event

# Create event bus
event_bus = EventBus()

# Subscribe to events
def handle_new_tick(event: Event):
    tick_data = event.payload
    print(f"New tick: {tick_data.symbol} @ {tick_data.price}")

event_bus.subscribe(EventType.NEW_TICK, handle_new_tick)

# Publish events
event = event_bus.create_event(
    EventType.NEW_TICK,
    tick_data,
    "market_data_handler"
)
await event_bus.publish(event)
```

### Event Bus (`event_bus.py`)

Enhanced event bus implementation with monitoring and performance optimization.

**Features:**
- Worker thread pool for parallel processing
- Event queue management with size limits
- Performance metrics collection
- Circuit breaker for error handling

**Configuration:**
```python
event_bus_config = {
    'worker_threads': 4,
    'queue_size': 10000,
    'enable_metrics': True,
    'circuit_breaker_threshold': 10
}
```

### Component Base (`component_base.py`)

Base class providing standard component functionality.

**Features:**
- Standardized component lifecycle (initialize → start → stop)
- Health monitoring and status reporting
- Performance metrics collection
- State persistence and recovery
- Event subscription management

**Usage:**
```python
from src.core.component_base import ComponentBase

class MyComponent(ComponentBase):
    def _component_initialize(self) -> bool:
        # Component-specific initialization
        return True
    
    def _component_start(self) -> None:
        # Start component operations
        pass
    
    def _component_stop(self) -> None:
        # Stop component operations
        pass
    
    def save_state(self) -> None:
        # Save component state
        pass
    
    def load_state(self) -> None:
        # Load component state
        pass
```

### Configuration Management (`minimal_config.py`)

Lightweight configuration loading and validation.

**Features:**
- YAML configuration file support
- Environment variable substitution
- Configuration validation
- Hot-reloading support

**Usage:**
```python
from src.core.minimal_config import load_config

# Load configuration with environment variable substitution
config = load_config("configs/production.yaml")

# Access configuration values
symbols = config["data_handler"]["symbols"]
marl_enabled = config["strategic_marl"]["enabled"]
```

**Environment Variables:**
```bash
export DATABASE_URL="postgresql://user:pass@localhost/db"
export REDIS_URL="redis://localhost:6379"
export MARKET_DATA_API_KEY="your_api_key_here"
```

**Configuration Example:**
```yaml
database:
  url: ${DATABASE_URL}
  pool_size: 10

redis:
  url: ${REDIS_URL}
  max_connections: 100

market_data:
  api_key: ${MARKET_DATA_API_KEY}
  symbols: ["ES", "NQ", "YM"]
```

## Performance Considerations

### Event Bus Optimization

The event bus is optimized for high-frequency trading:

- **Batched Processing**: Events are processed in batches for better throughput
- **Worker Threads**: Multiple worker threads handle event processing
- **Memory Management**: Circular buffers prevent memory leaks
- **Circuit Breaker**: Automatic error recovery prevents cascading failures

### Component Lifecycle

Components follow a strict lifecycle to ensure proper initialization:

1. **Instantiation**: Create component instances
2. **Dependency Resolution**: Calculate initialization order
3. **Initialization**: Initialize components in dependency order
4. **Event Wiring**: Connect components via event subscriptions
5. **Startup**: Start component operations
6. **Runtime**: Normal operation with health monitoring
7. **Shutdown**: Graceful shutdown in reverse order

### Memory Management

- Components implement proper cleanup in `save_state()`
- Event bus uses bounded queues to prevent memory buildup
- Circular references are avoided in component design
- Garbage collection is optimized for low-latency operation

## Error Handling

### Component Isolation

Each component is isolated to prevent failures from cascading:

```python
try:
    component.process_data(data)
except Exception as e:
    logger.error(f"Component {component.name} error: {e}")
    # Other components continue operating
```

### Event Bus Resilience

The event bus handles subscriber errors gracefully:

- Exceptions in event handlers don't interrupt other handlers
- Failed events are logged with full context
- Circuit breaker prevents overwhelming failing components
- Dead letter queue for unprocessable events

### Recovery Mechanisms

- Components can be restarted individually
- State persistence enables recovery from failures
- Health checks trigger automatic recovery
- Graceful degradation when components are unavailable

## Monitoring and Observability

### Health Checks

Each component provides health status:

```python
health_status = component.get_health_status()
# Returns:
# {
#     'name': 'ComponentName',
#     'state': 'running',
#     'uptime': 3600.0,
#     'processed_count': 10000,
#     'error_count': 2,
#     'error_rate': 0.0002,
#     'checks': {
#         'database_connection': {'status': 'healthy'},
#         'memory_usage': {'status': 'healthy', 'value': '512MB'}
#     }
# }
```

### Performance Metrics

Components collect standardized metrics:

```python
metrics = component.get_metrics()
# Returns:
# {
#     'uptime': 3600.0,
#     'processed_count': 10000,
#     'error_count': 2,
#     'error_rate': 0.0002,
#     'average_processing_time': 0.001,
#     'memory_usage': 512.0,
#     'cpu_usage': 15.5
# }
```

### Event Bus Metrics

Event bus provides detailed performance information:

```python
event_stats = event_bus.get_event_statistics()
# Returns:
# {
#     'queue_size': 150,
#     'active_workers': 4,
#     'event_stats': {
#         'NEW_TICK': {'published': 50000, 'processed': 49999, 'failed': 1},
#         'NEW_BAR': {'published': 500, 'processed': 500, 'failed': 0}
#     },
#     'performance_metrics': {
#         'avg_processing_time': 0.0005,
#         'throughput_per_second': 8500
#     }
# }
```

## Configuration Examples

### Development Configuration

```yaml
system:
  environment: development
  log_level: DEBUG
  debug: true

event_bus:
  worker_threads: 2
  queue_size: 1000
  enable_metrics: true

logging:
  level: DEBUG
  handlers:
    - type: console
    - type: file
      filename: logs/grandmodel_dev.log
```

### Production Configuration

```yaml
system:
  environment: production
  log_level: INFO
  debug: false

event_bus:
  worker_threads: 8
  queue_size: 100000
  enable_metrics: true
  circuit_breaker_threshold: 50

performance:
  cpu_affinity: [0, 1, 2, 3]
  memory_limit: 8G
  gc_frequency: 300

monitoring:
  health_check_interval: 30
  metrics_export_interval: 60
  prometheus_port: 9090
```

### High-Performance Configuration

```yaml
system:
  environment: production
  log_level: WARNING  # Reduced logging for performance

event_bus:
  worker_threads: 16
  queue_size: 1000000
  batch_size: 1000
  enable_metrics: false  # Disable for maximum performance

performance:
  cpu_affinity: [0, 1, 2, 3, 4, 5, 6, 7]
  memory_limit: 16G
  huge_pages: true
  numa_affinity: 0

optimization:
  jit_compilation: true
  vectorized_operations: true
  memory_pooling: true
```

## Best Practices

### Component Development

1. **Inherit from ComponentBase**: Use the standard base class for consistency
2. **Implement Required Methods**: All abstract methods must be implemented
3. **Handle Errors Gracefully**: Use try-catch blocks and proper logging
4. **Validate Configuration**: Check required configuration fields
5. **Monitor Performance**: Track metrics and processing times

### Event Usage

1. **Use Appropriate Event Types**: Choose the correct event type for your use case
2. **Keep Payloads Small**: Large payloads impact performance
3. **Handle Events Asynchronously**: Don't block event handlers
4. **Validate Event Data**: Check payload structure and content
5. **Implement Backpressure**: Handle high-frequency event scenarios

### Configuration Management

1. **Use Environment Variables**: For secrets and environment-specific values
2. **Validate Configuration**: Check required fields and value ranges
3. **Document Configuration**: Provide clear documentation for all options
4. **Version Configuration**: Track configuration changes
5. **Test Configuration**: Validate configurations in CI/CD

## Testing

### Unit Tests

```python
# tests/unit/test_core/test_component_base.py
import unittest
from unittest.mock import MagicMock
from src.core.component_base import ComponentBase

class TestComponent(ComponentBase):
    def _component_initialize(self) -> bool:
        return True
    
    def _component_start(self) -> None:
        pass
    
    def _component_stop(self) -> None:
        pass
    
    def save_state(self) -> None:
        pass
    
    def load_state(self) -> None:
        pass

class TestComponentBase(unittest.TestCase):
    def setUp(self):
        self.mock_event_bus = MagicMock()
        self.component = TestComponent("test_component", {}, self.mock_event_bus)
    
    def test_component_lifecycle(self):
        # Test initialization
        self.assertTrue(self.component.initialize())
        
        # Test startup
        self.component.start()
        
        # Test health status
        health = self.component.get_health_status()
        self.assertEqual(health['name'], 'test_component')
        self.assertEqual(health['state'], 'running')
        
        # Test shutdown
        self.component.stop()
        self.assertEqual(self.component._state, ComponentState.STOPPED)
```

### Integration Tests

```python
# tests/integration/test_core/test_kernel_integration.py
import pytest
from src.core.kernel import AlgoSpaceKernel

@pytest.mark.integration
class TestKernelIntegration:
    def test_full_system_startup(self):
        # Test complete system initialization
        kernel = AlgoSpaceKernel("configs/test.yaml")
        
        # Initialize and start
        success = kernel.initialize()
        assert success
        
        # Verify components are loaded
        status = kernel.get_status()
        assert status['running']
        assert len(status['components']) > 0
        
        # Cleanup
        kernel.shutdown()
```

## Troubleshooting

### Common Issues

**Component Initialization Failures:**
- Check configuration file syntax and required fields
- Verify dependency order in component definitions
- Review error logs for specific failure reasons

**Event Bus Performance Issues:**
- Monitor queue sizes and worker thread utilization
- Check for slow event handlers blocking processing
- Adjust worker thread count based on load

**Memory Leaks:**
- Verify components implement proper cleanup
- Check for circular references in event subscriptions
- Monitor garbage collection frequency and effectiveness

### Debug Commands

```bash
# Check component health
curl http://localhost:8000/health

# View system metrics
curl http://localhost:8000/metrics

# Check event bus statistics
curl http://localhost:8000/events/stats

# View component dependencies
python -c "from src.core.kernel import AlgoSpaceKernel; k=AlgoSpaceKernel('config.yaml'); print(k.get_dependency_graph())"
```

### Performance Tuning

```python
# Enable debug mode for detailed performance information
config = {
    'system': {'debug': True, 'log_level': 'DEBUG'},
    'event_bus': {'enable_metrics': True},
    'performance': {'profile_components': True}
}

# Profile component performance
kernel = AlgoSpaceKernel(config)
performance_report = kernel.get_performance_report()
```

## Related Documentation

- [Architecture Overview](../../docs/architecture/system_overview.md)
- [Component Design](../../docs/architecture/component_design.md)
- [Event System Guide](../../docs/api/events_api.md)
- [Configuration Reference](../../docs/guides/configuration_guide.md)