# Kernel API Documentation

## Overview

The `AlgoSpaceKernel` is the central orchestration component of the GrandModel trading system. It manages the lifecycle of all system components, handles configuration loading, and coordinates communication through the event bus.

## Table of Contents

- [Class: AlgoSpaceKernel](#class-algospacekernel)
- [Configuration](#configuration)
- [Component Management](#component-management)
- [Event System Integration](#event-system-integration)
- [Error Handling](#error-handling)
- [Examples](#examples)

## Class: AlgoSpaceKernel

### Constructor

```python
class AlgoSpaceKernel:
    def __init__(self, config_path: str = "config/settings.yaml")
```

**Parameters:**
- `config_path` (str): Path to the system configuration file. Defaults to "config/settings.yaml"

**Description:**
Initializes the kernel with the specified configuration. The kernel is created in an uninitialized state and requires calling `initialize()` before use.

### Methods

#### initialize()

```python
def initialize(self) -> None
```

**Description:**
Initializes and wires all system components in the correct dependency order. This method must be called before `run()`.

**Phases:**
1. **Component Instantiation**: Creates all configured components
2. **Event Wiring**: Connects components via event subscriptions
3. **Component Initialization**: Performs post-instantiation setup

**Raises:**
- `Exception`: If initialization fails at any stage

**Example:**
```python
kernel = AlgoSpaceKernel("config/production.yaml")
kernel.initialize()
```

#### run()

```python
def run(self) -> None
```

**Description:**
Starts the main system loop. This method blocks and runs until a shutdown is requested.

**Process:**
1. Starts data streams
2. Begins event dispatcher
3. Handles graceful shutdown on interruption

**Raises:**
- `RuntimeError`: If kernel not initialized
- `KeyboardInterrupt`: On user interruption (handled gracefully)

**Example:**
```python
try:
    kernel.run()
except KeyboardInterrupt:
    print("System shutdown requested")
```

#### shutdown()

```python
def shutdown(self) -> None
```

**Description:**
Initiates a graceful shutdown of the entire system.

**Shutdown Process:**
1. Stops data streams
2. Closes all open positions
3. Saves component states
4. Stops event bus
5. Logs shutdown completion

**Example:**
```python
# Programmatic shutdown
kernel.shutdown()
```

#### get_component()

```python
def get_component(self, name: str) -> Optional[Any]
```

**Parameters:**
- `name` (str): The component name

**Returns:**
- The component instance or `None` if not found

**Example:**
```python
data_handler = kernel.get_component("data_handler")
if data_handler:
    print("Data handler is available")
```

#### get_status()

```python
def get_status(self) -> Dict[str, Any]
```

**Returns:**
Dictionary containing system status information:
- `running` (bool): Whether the system is currently running
- `mode` (str): Operating mode (live, backtest, etc.)
- `components` (List[str]): List of active component names
- `subscribers` (int): Number of event subscribers

**Example:**
```python
status = kernel.get_status()
print(f"System running: {status['running']}")
print(f"Active components: {status['components']}")
```

#### get_event_bus()

```python
def get_event_bus(self) -> EventBus
```

**Returns:**
The system's `EventBus` instance for direct event handling.

**Example:**
```python
event_bus = kernel.get_event_bus()
event_bus.publish(custom_event)
```

## Configuration

The kernel uses YAML configuration files to define system behavior. Key configuration sections:

### Data Handler Configuration

```yaml
data_handler:
  type: "rithmic"  # or "ib", "backtest"
  connection:
    host: "localhost"
    port: 3001
  symbols: ["ES", "NQ"]
```

### Matrix Assemblers Configuration

```yaml
matrix_assemblers:
  30m:
    window_size: 48
    features:
      - "mlmi_value"
      - "mlmi_signal"
      - "nwrqk_value"
      - "time_hour_sin"
  5m:
    window_size: 60
    features:
      - "fvg_bullish_active"
      - "fvg_bearish_active"
      - "price_momentum_5"
```

### Strategic MARL Configuration

```yaml
strategic_marl:
  enabled: true
  model_path: "models/strategic_agent.pth"
  learning_rate: 0.0001
  batch_size: 32
```

## Component Management

The kernel manages the following component categories:

### Data Pipeline Components
- **LiveDataHandler**: Real-time market data processing
- **BacktestDataHandler**: Historical data simulation
- **BarGenerator**: Time-based bar aggregation

### Analysis Components
- **IndicatorEngine**: Technical indicator calculations
- **MatrixAssembler30m**: Strategic feature matrices
- **MatrixAssembler5m**: Tactical feature matrices
- **SynergyDetector**: Pattern recognition

### Intelligence Components
- **StrategicMARLComponent**: Strategic decision making
- **RDEComponent**: Regime detection
- **MRMSComponent**: Multi-resolution market state
- **MainMARLCoreComponent**: Core MARL coordination

### Execution Components
- **LiveExecutionHandler**: Real-time order execution
- **BacktestExecutionHandler**: Simulated order execution

## Event System Integration

The kernel automatically wires components through the event system:

### Data Flow Events
```python
# Tick data → Bar generation
EventType.NEW_TICK → BarGenerator.on_new_tick

# Bar data → Indicator calculation
EventType.NEW_5MIN_BAR → IndicatorEngine.on_new_bar
EventType.NEW_30MIN_BAR → IndicatorEngine.on_new_bar

# Indicators → Matrix assembly
EventType.INDICATORS_READY → MatrixAssembler.on_indicators_ready
```

### Decision Flow Events
```python
# Synergy detection → MARL decision
EventType.SYNERGY_DETECTED → StrategicMARLComponent.handle_synergy

# MARL decision → Execution
EventType.EXECUTE_TRADE → ExecutionHandler.execute_trade
```

### Feedback Events
```python
# Trade completion → Learning feedback
EventType.TRADE_CLOSED → MainMARLCore.record_outcome
```

## Error Handling

### System Error Handling

The kernel implements comprehensive error handling:

```python
def _handle_system_error(self, error_info: Dict[str, Any]) -> None:
    """Handle system-wide errors"""
    if error_info.get("critical", False):
        logger.critical("Critical error detected. Initiating shutdown.")
        self.shutdown()
```

### Component Error Isolation

Individual component failures don't crash the entire system:

```python
try:
    component.process_data(data)
except Exception as e:
    logger.error(f"Component {component.name} error: {e}")
    # System continues running
```

### Graceful Degradation

The system can operate with missing components:

```python
if "strategic_marl" in self.components:
    # Use MARL for decisions
    pass
else:
    # Fall back to traditional indicators
    logger.warning("MARL not available, using fallback logic")
```

## Examples

### Basic Usage

```python
from src.core.kernel import AlgoSpaceKernel

# Initialize kernel
kernel = AlgoSpaceKernel("config/production.yaml")

try:
    # Initialize all components
    kernel.initialize()
    
    # Check system status
    status = kernel.get_status()
    print(f"Components loaded: {status['components']}")
    
    # Start the system
    kernel.run()
    
except Exception as e:
    print(f"System error: {e}")
    kernel.shutdown()
```

### Custom Component Access

```python
# Get specific components
data_handler = kernel.get_component("data_handler")
strategic_marl = kernel.get_component("strategic_marl")

# Check component availability
if strategic_marl:
    print("Strategic MARL is active")
    
# Access event bus for custom events
event_bus = kernel.get_event_bus()
custom_event = event_bus.create_event(
    EventType.CUSTOM_SIGNAL,
    {"signal": "strong_bullish"},
    "custom_strategy"
)
event_bus.publish(custom_event)
```

### Production Monitoring

```python
import time
import logging

kernel = AlgoSpaceKernel("config/production.yaml")
kernel.initialize()

# Start system in background thread
import threading
system_thread = threading.Thread(target=kernel.run)
system_thread.start()

# Monitor system health
while system_thread.is_alive():
    status = kernel.get_status()
    logging.info(f"System status: {status}")
    time.sleep(60)  # Check every minute
```

### Configuration Validation

```python
def validate_kernel_config(config_path: str) -> bool:
    """Validate kernel configuration before startup"""
    try:
        kernel = AlgoSpaceKernel(config_path)
        # This loads and validates the configuration
        config = kernel.config
        
        # Check required sections
        required_sections = ["data_handler", "matrix_assemblers"]
        for section in required_sections:
            if section not in config:
                print(f"Missing required config section: {section}")
                return False
                
        return True
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        return False

# Usage
if validate_kernel_config("config/production.yaml"):
    kernel = AlgoSpaceKernel("config/production.yaml")
    kernel.initialize()
    kernel.run()
```

## Performance Considerations

### Memory Management
- The kernel maintains references to all components
- Components should implement proper cleanup in `save_state()`
- Use `get_component()` sparingly in performance-critical paths

### Event Bus Performance
- Event publishing is synchronous by default
- High-frequency events (ticks, bars) are logged at debug level only
- Consider event batching for very high-frequency scenarios

### Initialization Time
- Component initialization happens sequentially
- Model loading can be time-intensive
- Consider implementing health checks for production deployment

## Thread Safety

The kernel is designed for single-threaded operation with the event bus handling asynchronous patterns. For multi-threaded scenarios:

1. Use thread-safe event publishing
2. Ensure component thread safety
3. Consider using asyncio for I/O-bound operations

## Best Practices

1. **Always call `initialize()` before `run()`**
2. **Handle exceptions in the main thread**
3. **Use configuration files for all settings**
4. **Monitor system status in production**
5. **Implement proper logging for debugging**
6. **Test configuration changes in development first**

## Troubleshooting

### Common Issues

**ImportError for components:**
```bash
# Check if all dependencies are installed
pip install -r requirements.txt

# Verify Python path
export PYTHONPATH=/path/to/GrandModel
```

**Configuration not found:**
```python
# Use absolute paths for configuration
kernel = AlgoSpaceKernel("/full/path/to/config.yaml")
```

**Component initialization failures:**
```python
# Check logs for specific component errors
tail -f logs/grandmodel.log
```

### Debug Mode

Enable debug logging for detailed information:

```python
import logging
logging.getLogger().setLevel(logging.DEBUG)

kernel = AlgoSpaceKernel("config/debug.yaml")
```

## Related Documentation

- [Event System API](events_api.md)
- [Component Development Guide](../development/component_guide.md)
- [Configuration Reference](../guides/configuration_guide.md)
- [Deployment Guide](../deployment/production_deployment.md)