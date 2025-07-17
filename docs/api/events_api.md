# Event System API Documentation

## Overview

The GrandModel event system provides a robust, type-safe messaging infrastructure that enables loose coupling between system components. All inter-component communication flows through the centralized `EventBus`, ensuring scalability and maintainability.

## Table of Contents

- [Core Classes](#core-classes)
- [Event Types](#event-types)
- [Data Structures](#data-structures)
- [EventBus API](#eventbus-api)
- [Event Publishing](#event-publishing)
- [Event Subscription](#event-subscription)
- [Performance Considerations](#performance-considerations)
- [Examples](#examples)

## Core Classes

### EventBus

The central event dispatch system that manages all event routing in the GrandModel system.

```python
class EventBus:
    def __init__(self)
    def subscribe(self, event_type: EventType, callback: Callable[[Event], None]) -> None
    def unsubscribe(self, event_type: EventType, callback: Callable[[Event], None]) -> None
    def publish(self, event: Event) -> None
    def create_event(self, event_type: EventType, payload: Any, source: str) -> Event
    def dispatch_forever(self) -> None
    def stop(self) -> None
```

### Event

Base event structure that carries information between components.

```python
@dataclass
class Event:
    event_type: EventType
    timestamp: datetime
    payload: Any
    source: str
```

## Event Types

The system defines comprehensive event types organized by functional category:

### Data Events

```python
class EventType(Enum):
    # Raw market data
    NEW_TICK = "NEW_TICK"
    NEW_5MIN_BAR = "NEW_5MIN_BAR"
    NEW_30MIN_BAR = "NEW_30MIN_BAR"
    NEW_BAR = "NEW_BAR"  # Generic bar event
```

**Usage:**
- `NEW_TICK`: Real-time price updates from market data feeds
- `NEW_5MIN_BAR`: Tactical timeframe bar completion
- `NEW_30MIN_BAR`: Strategic timeframe bar completion
- `NEW_BAR`: Legacy support for generic bar events

### System Events

```python
    # System lifecycle
    SYSTEM_START = "SYSTEM_START"
    SYSTEM_SHUTDOWN = "SYSTEM_SHUTDOWN"
    BACKTEST_COMPLETE = "BACKTEST_COMPLETE"
    SYSTEM_ERROR = "SYSTEM_ERROR"
    COMPONENT_STARTED = "COMPONENT_STARTED"
    COMPONENT_STOPPED = "COMPONENT_STOPPED"
    SHUTDOWN_REQUEST = "SHUTDOWN_REQUEST"
```

**Usage:**
- `SYSTEM_START`: System initialization complete
- `SYSTEM_SHUTDOWN`: Graceful shutdown initiated
- `SYSTEM_ERROR`: Critical system errors requiring attention
- `SHUTDOWN_REQUEST`: Request for graceful system shutdown

### Connection Events

```python
    # Network and connectivity
    CONNECTION_LOST = "CONNECTION_LOST"
    CONNECTION_RESTORED = "CONNECTION_RESTORED"
```

**Usage:**
- Monitor broker connection health
- Trigger reconnection logic
- Implement connection-dependent component behavior

### Analysis Events

```python
    # Technical analysis
    INDICATOR_UPDATE = "INDICATOR_UPDATE"
    INDICATORS_READY = "INDICATORS_READY"
    SYNERGY_DETECTED = "SYNERGY_DETECTED"
```

**Usage:**
- `INDICATOR_UPDATE`: Individual indicator calculation complete
- `INDICATORS_READY`: All indicators for a timeframe are ready
- `SYNERGY_DETECTED`: Multi-indicator pattern recognition

### MARL Events

```python
    # Multi-Agent Reinforcement Learning
    TRADE_QUALIFIED = "TRADE_QUALIFIED"
    TRADE_REJECTED = "TRADE_REJECTED"
    EXECUTE_TRADE = "EXECUTE_TRADE"
    STRATEGIC_DECISION = "STRATEGIC_DECISION"
```

**Usage:**
- `TRADE_QUALIFIED`: MARL agents approve a trading opportunity
- `TRADE_REJECTED`: MARL agents reject a trading opportunity
- `EXECUTE_TRADE`: Final decision to execute a trade
- `STRATEGIC_DECISION`: High-level strategic direction change

### Execution Events

```python
    # Order management
    ORDER_SUBMITTED = "ORDER_SUBMITTED"
    ORDER_FILLED = "ORDER_FILLED"
    ORDER_CANCELLED = "ORDER_CANCELLED"
    TRADE_CLOSED = "TRADE_CLOSED"
```

**Usage:**
- Track order lifecycle from submission to completion
- Trigger post-execution logic and learning updates
- Maintain audit trail for compliance

### Risk Events

```python
    # Risk management
    RISK_BREACH = "RISK_BREACH"
    RISK_UPDATE = "RISK_UPDATE"
    VAR_UPDATE = "VAR_UPDATE"
    KELLY_SIZING = "KELLY_SIZING"
    POSITION_UPDATE = "POSITION_UPDATE"
```

**Usage:**
- `RISK_BREACH`: Risk limits exceeded, immediate action required
- `VAR_UPDATE`: Value-at-Risk calculation update
- `KELLY_SIZING`: Optimal position size calculation
- `POSITION_UPDATE`: Portfolio position changes

## Data Structures

### TickData

```python
@dataclass
class TickData:
    symbol: str
    timestamp: datetime
    price: float
    volume: int
```

**Example:**
```python
tick = TickData(
    symbol="ES",
    timestamp=datetime.now(),
    price=4250.75,
    volume=100
)
```

### BarData

```python
@dataclass
class BarData:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    timeframe: int  # 5 or 30 (minutes)
```

**Example:**
```python
bar = BarData(
    symbol="ES",
    timestamp=datetime.now(),
    open=4250.00,
    high=4252.50,
    low=4249.25,
    close=4251.75,
    volume=1500,
    timeframe=5
)
```

## EventBus API

### Constructor

```python
def __init__(self):
    """Initialize the event bus with empty subscriber registry"""
```

### Subscription Management

#### subscribe()

```python
def subscribe(self, event_type: EventType, callback: Callable[[Event], None]) -> None
```

**Parameters:**
- `event_type`: The type of event to subscribe to
- `callback`: Function to call when event is published

**Example:**
```python
def handle_new_tick(event: Event):
    tick_data = event.payload
    print(f"New tick: {tick_data.symbol} @ {tick_data.price}")

event_bus.subscribe(EventType.NEW_TICK, handle_new_tick)
```

#### unsubscribe()

```python
def unsubscribe(self, event_type: EventType, callback: Callable[[Event], None]) -> None
```

**Parameters:**
- `event_type`: The event type to unsubscribe from
- `callback`: The exact callback function to remove

**Example:**
```python
event_bus.unsubscribe(EventType.NEW_TICK, handle_new_tick)
```

### Event Publishing

#### publish()

```python
def publish(self, event: Event) -> None
```

**Parameters:**
- `event`: The event instance to publish

**Description:**
Publishes an event to all registered subscribers. Exceptions in callbacks are caught and logged but don't interrupt other callbacks.

**Example:**
```python
tick_event = event_bus.create_event(
    EventType.NEW_TICK,
    TickData("ES", datetime.now(), 4250.75, 100),
    "market_data_feed"
)
event_bus.publish(tick_event)
```

#### create_event()

```python
def create_event(self, event_type: EventType, payload: Any, source: str) -> Event
```

**Parameters:**
- `event_type`: Type of event to create
- `payload`: Event data/content
- `source`: Component name that created the event

**Returns:**
- `Event`: New event instance with current timestamp

**Example:**
```python
# Create a custom event
risk_event = event_bus.create_event(
    EventType.RISK_BREACH,
    {"limit": "VAR", "current": 0.05, "max": 0.03},
    "risk_manager"
)
```

### Event Loop Management

#### dispatch_forever()

```python
def dispatch_forever(self) -> None
```

**Description:**
Starts the main event dispatch loop. This method blocks until interrupted by KeyboardInterrupt.

**Example:**
```python
try:
    event_bus.dispatch_forever()
except KeyboardInterrupt:
    print("Event loop interrupted")
```

#### stop()

```python
def stop(self) -> None
```

**Description:**
Stops the event bus and cleans up resources.

## Event Publishing

### Basic Publishing

```python
# Method 1: Create and publish separately
event = event_bus.create_event(
    EventType.INDICATOR_UPDATE,
    {"indicator": "RSI", "value": 65.5},
    "indicator_engine"
)
event_bus.publish(event)

# Method 2: Direct Event creation
from datetime import datetime
from src.core.events import Event

direct_event = Event(
    event_type=EventType.SYNERGY_DETECTED,
    timestamp=datetime.now(),
    payload={"pattern": "TYPE_1", "confidence": 0.85},
    source="synergy_detector"
)
event_bus.publish(direct_event)
```

### High-Frequency Events

For high-frequency events like ticks, consider batching:

```python
def publish_tick_batch(event_bus, ticks):
    """Publish multiple ticks efficiently"""
    for tick in ticks:
        event = event_bus.create_event(
            EventType.NEW_TICK,
            tick,
            "data_handler"
        )
        event_bus.publish(event)
```

### Error Handling in Publishing

```python
def safe_publish(event_bus, event_type, payload, source):
    """Safely publish an event with error handling"""
    try:
        event = event_bus.create_event(event_type, payload, source)
        event_bus.publish(event)
        return True
    except Exception as e:
        logger.error(f"Failed to publish event {event_type}: {e}")
        return False
```

## Event Subscription

### Basic Subscription

```python
class MyComponent:
    def __init__(self, event_bus):
        self.event_bus = event_bus
        self._setup_subscriptions()
    
    def _setup_subscriptions(self):
        """Register for events of interest"""
        self.event_bus.subscribe(EventType.NEW_5MIN_BAR, self.on_new_bar)
        self.event_bus.subscribe(EventType.INDICATORS_READY, self.on_indicators_ready)
    
    def on_new_bar(self, event: Event):
        """Handle new bar data"""
        bar_data = event.payload
        print(f"Processing {bar_data.timeframe}min bar for {bar_data.symbol}")
    
    def on_indicators_ready(self, event: Event):
        """Handle indicator completion"""
        indicator_data = event.payload
        print(f"Indicators ready: {indicator_data}")
```

### Conditional Event Handling

```python
def on_new_tick(self, event: Event):
    """Handle tick data with filtering"""
    tick = event.payload
    
    # Only process ES contract ticks
    if tick.symbol != "ES":
        return
    
    # Only process during market hours
    if not self.is_market_hours(tick.timestamp):
        return
    
    # Process the relevant tick
    self.process_es_tick(tick)
```

### Multiple Event Subscriptions

```python
class TradingStrategy:
    def __init__(self, event_bus):
        self.event_bus = event_bus
        self.setup_event_handlers()
    
    def setup_event_handlers(self):
        """Subscribe to multiple related events"""
        events_and_handlers = [
            (EventType.SYNERGY_DETECTED, self.on_synergy),
            (EventType.RISK_UPDATE, self.on_risk_update),
            (EventType.TRADE_CLOSED, self.on_trade_closed),
            (EventType.CONNECTION_LOST, self.on_connection_lost)
        ]
        
        for event_type, handler in events_and_handlers:
            self.event_bus.subscribe(event_type, handler)
```

### Unsubscription and Cleanup

```python
class ComponentWithCleanup:
    def __init__(self, event_bus):
        self.event_bus = event_bus
        self.handlers = []
        self.setup_handlers()
    
    def setup_handlers(self):
        """Track handlers for later cleanup"""
        handler_configs = [
            (EventType.NEW_TICK, self.on_tick),
            (EventType.NEW_BAR, self.on_bar)
        ]
        
        for event_type, handler in handler_configs:
            self.event_bus.subscribe(event_type, handler)
            self.handlers.append((event_type, handler))
    
    def cleanup(self):
        """Clean up all subscriptions"""
        for event_type, handler in self.handlers:
            self.event_bus.unsubscribe(event_type, handler)
        self.handlers.clear()
```

## Performance Considerations

### Event Frequency

The system handles different event frequencies appropriately:

- **High-frequency** (NEW_TICK): Logged at debug level only
- **Medium-frequency** (NEW_BAR): Standard info logging
- **Low-frequency** (SYSTEM_EVENTS): Full logging with details

### Memory Management

```python
# Avoid holding references to large payloads
def efficient_handler(self, event: Event):
    """Process event data without holding references"""
    payload = event.payload
    
    # Extract only needed data
    symbol = payload.symbol
    price = payload.price
    
    # Don't store the entire payload
    self.process_price_update(symbol, price)
    # payload can be garbage collected
```

### Asynchronous Considerations

While the event bus is synchronous, handlers can be made asynchronous:

```python
import asyncio

class AsyncComponent:
    def __init__(self, event_bus):
        self.event_bus = event_bus
        self.event_bus.subscribe(EventType.NEW_TICK, self.on_tick)
    
    def on_tick(self, event: Event):
        """Synchronous event handler that delegates to async logic"""
        # Don't block the event bus
        asyncio.create_task(self.async_process_tick(event.payload))
    
    async def async_process_tick(self, tick_data):
        """Asynchronous tick processing"""
        # Perform I/O or CPU-intensive operations
        await self.update_database(tick_data)
        await self.notify_external_service(tick_data)
```

## Examples

### Complete Component Example

```python
import logging
from typing import Dict, Any
from src.core.events import EventBus, EventType, Event

class IndicatorEngine:
    """Example component using the event system"""
    
    def __init__(self, config: Dict[str, Any], event_bus: EventBus):
        self.config = config
        self.event_bus = event_bus
        self.logger = logging.getLogger(self.__class__.__name__)
        self.indicators = {}
        
        # Subscribe to relevant events
        self._setup_event_subscriptions()
    
    def _setup_event_subscriptions(self):
        """Register for events this component needs"""
        self.event_bus.subscribe(EventType.NEW_5MIN_BAR, self.on_new_bar)
        self.event_bus.subscribe(EventType.NEW_30MIN_BAR, self.on_new_bar)
        self.event_bus.subscribe(EventType.SYSTEM_SHUTDOWN, self.on_shutdown)
    
    def on_new_bar(self, event: Event):
        """Process new bar data and calculate indicators"""
        bar_data = event.payload
        
        try:
            # Calculate indicators for this bar
            indicators = self._calculate_indicators(bar_data)
            
            # Update internal state
            self.indicators[bar_data.symbol] = indicators
            
            # Publish indicator update
            self._publish_indicator_update(bar_data.symbol, indicators)
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")
            self._publish_error(e)
    
    def _calculate_indicators(self, bar_data) -> Dict[str, float]:
        """Calculate technical indicators"""
        # Implementation would go here
        return {
            "rsi": 65.5,
            "macd": 0.25,
            "bollinger_upper": bar_data.close * 1.02
        }
    
    def _publish_indicator_update(self, symbol: str, indicators: Dict[str, float]):
        """Publish calculated indicators"""
        event = self.event_bus.create_event(
            EventType.INDICATOR_UPDATE,
            {
                "symbol": symbol,
                "indicators": indicators,
                "timestamp": datetime.now()
            },
            self.__class__.__name__
        )
        self.event_bus.publish(event)
    
    def _publish_error(self, error: Exception):
        """Publish system error"""
        event = self.event_bus.create_event(
            EventType.SYSTEM_ERROR,
            {
                "component": self.__class__.__name__,
                "error": str(error),
                "critical": False
            },
            self.__class__.__name__
        )
        self.event_bus.publish(event)
    
    def on_shutdown(self, event: Event):
        """Handle graceful shutdown"""
        self.logger.info("Indicator engine shutting down")
        # Save state, cleanup resources, etc.
```

### Event Flow Example

```python
# Typical event flow in the GrandModel system
def demonstrate_event_flow():
    event_bus = EventBus()
    
    # 1. Market data arrives
    tick_data = TickData("ES", datetime.now(), 4250.75, 100)
    tick_event = event_bus.create_event(
        EventType.NEW_TICK, 
        tick_data, 
        "market_data_feed"
    )
    event_bus.publish(tick_event)
    
    # 2. Bar completes (triggered by bar generator)
    bar_data = BarData("ES", datetime.now(), 4250.0, 4252.0, 4249.0, 4251.0, 1500, 5)
    bar_event = event_bus.create_event(
        EventType.NEW_5MIN_BAR,
        bar_data,
        "bar_generator"
    )
    event_bus.publish(bar_event)
    
    # 3. Indicators calculated (triggered by indicator engine)
    indicators = {"rsi": 65.5, "macd": 0.25}
    indicator_event = event_bus.create_event(
        EventType.INDICATORS_READY,
        {"symbol": "ES", "indicators": indicators},
        "indicator_engine"
    )
    event_bus.publish(indicator_event)
    
    # 4. Synergy detected (triggered by synergy detector)
    synergy_event = event_bus.create_event(
        EventType.SYNERGY_DETECTED,
        {"pattern": "TYPE_1", "confidence": 0.85},
        "synergy_detector"
    )
    event_bus.publish(synergy_event)
    
    # 5. Trade decision (triggered by MARL component)
    trade_event = event_bus.create_event(
        EventType.EXECUTE_TRADE,
        {
            "symbol": "ES",
            "direction": "long",
            "quantity": 2,
            "price": 4251.0
        },
        "strategic_marl"
    )
    event_bus.publish(trade_event)
```

### Testing Event System

```python
import unittest
from unittest.mock import Mock, patch

class TestEventSystem(unittest.TestCase):
    def setUp(self):
        self.event_bus = EventBus()
        self.callback_mock = Mock()
    
    def test_subscription_and_publishing(self):
        """Test basic subscription and event publishing"""
        # Subscribe to event
        self.event_bus.subscribe(EventType.NEW_TICK, self.callback_mock)
        
        # Create and publish event
        tick_data = TickData("ES", datetime.now(), 4250.0, 100)
        event = self.event_bus.create_event(
            EventType.NEW_TICK,
            tick_data,
            "test_source"
        )
        self.event_bus.publish(event)
        
        # Verify callback was called
        self.callback_mock.assert_called_once()
        called_event = self.callback_mock.call_args[0][0]
        self.assertEqual(called_event.event_type, EventType.NEW_TICK)
        self.assertEqual(called_event.payload.symbol, "ES")
    
    def test_unsubscription(self):
        """Test event unsubscription"""
        # Subscribe and unsubscribe
        self.event_bus.subscribe(EventType.NEW_TICK, self.callback_mock)
        self.event_bus.unsubscribe(EventType.NEW_TICK, self.callback_mock)
        
        # Publish event
        event = self.event_bus.create_event(
            EventType.NEW_TICK,
            TickData("ES", datetime.now(), 4250.0, 100),
            "test_source"
        )
        self.event_bus.publish(event)
        
        # Verify callback was not called
        self.callback_mock.assert_not_called()
    
    def test_error_handling_in_callbacks(self):
        """Test that callback errors don't crash the event bus"""
        def failing_callback(event):
            raise Exception("Test error")
        
        def working_callback(event):
            self.callback_mock(event)
        
        # Subscribe both callbacks
        self.event_bus.subscribe(EventType.NEW_TICK, failing_callback)
        self.event_bus.subscribe(EventType.NEW_TICK, working_callback)
        
        # Publish event
        event = self.event_bus.create_event(
            EventType.NEW_TICK,
            TickData("ES", datetime.now(), 4250.0, 100),
            "test_source"
        )
        
        # Should not raise exception despite failing callback
        self.event_bus.publish(event)
        
        # Working callback should still be called
        self.callback_mock.assert_called_once()
```

## Related Documentation

- [Kernel API](kernel_api.md)
- [Component Development Guide](../development/component_guide.md)
- [System Architecture](../architecture/system_overview.md)
- [Performance Optimization](../guides/performance_guide.md)