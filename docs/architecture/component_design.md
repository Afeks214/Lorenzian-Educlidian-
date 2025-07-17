# Component Design Architecture

## Overview

GrandModel follows a component-based architecture where each component is a self-contained, loosely-coupled module with well-defined responsibilities. This design promotes modularity, testability, and maintainability while enabling horizontal scaling and independent deployment of components.

## Table of Contents

- [Component Design Principles](#component-design-principles)
- [Base Component Architecture](#base-component-architecture)
- [Core Components](#core-components)
- [Data Pipeline Components](#data-pipeline-components)
- [Analysis Components](#analysis-components)
- [Intelligence Components](#intelligence-components)
- [Execution Components](#execution-components)
- [Component Lifecycle Management](#component-lifecycle-management)
- [Inter-Component Communication](#inter-component-communication)
- [Component Testing Strategy](#component-testing-strategy)

## Component Design Principles

### Design Principles

1. **Single Responsibility**: Each component has one primary responsibility
2. **Loose Coupling**: Components interact through well-defined interfaces
3. **High Cohesion**: Related functionality is grouped within components
4. **Event-Driven**: Components communicate via events, not direct calls
5. **Stateless Design**: Components maintain minimal state for better scalability
6. **Error Isolation**: Component failures don't cascade to other components
7. **Configuration-Driven**: Component behavior is controlled by configuration
8. **Observable**: Components provide metrics and logging for monitoring

### Interface Contracts

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class ComponentInterface(ABC):
    """Base interface for all system components"""
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize component resources and connections"""
        pass
    
    @abstractmethod
    def start(self) -> None:
        """Start component operations"""
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """Stop component operations gracefully"""
        pass
    
    @abstractmethod
    def get_health_status(self) -> Dict[str, Any]:
        """Return component health and status information"""
        pass
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, float]:
        """Return component performance metrics"""
        pass
    
    @abstractmethod
    def save_state(self) -> None:
        """Save component state for recovery"""
        pass
    
    @abstractmethod
    def load_state(self) -> None:
        """Load component state from storage"""
        pass
```

## Base Component Architecture

### ComponentBase Class

```python
import logging
import time
import threading
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum

class ComponentState(Enum):
    UNINITIALIZED = "uninitialized"
    INITIALIZED = "initialized"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"

@dataclass
class ComponentMetrics:
    """Standard metrics for all components"""
    start_time: float
    last_update_time: float
    processed_count: int
    error_count: int
    average_processing_time: float
    memory_usage: float
    cpu_usage: float

class ComponentBase(ComponentInterface):
    """Base class for all system components"""
    
    def __init__(self, name: str, config: Dict[str, Any], event_bus: EventBus):
        self.name = name
        self.config = config
        self.event_bus = event_bus
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{name}")
        
        # Component state management
        self._state = ComponentState.UNINITIALIZED
        self._state_lock = threading.RLock()
        
        # Metrics tracking
        self.metrics = ComponentMetrics(
            start_time=time.time(),
            last_update_time=time.time(),
            processed_count=0,
            error_count=0,
            average_processing_time=0.0,
            memory_usage=0.0,
            cpu_usage=0.0
        )
        
        # Health monitoring
        self._health_checks: Dict[str, Callable] = {}
        self._register_default_health_checks()
        
        # Event subscriptions
        self._subscriptions: List[Tuple[EventType, Callable]] = []
    
    def initialize(self) -> bool:
        """Initialize component with error handling"""
        with self._state_lock:
            if self._state != ComponentState.UNINITIALIZED:
                self.logger.warning(f"Component {self.name} already initialized")
                return True
            
            self._state = ComponentState.INITIALIZED
        
        try:
            self.logger.info(f"Initializing component {self.name}")
            
            # Validate configuration
            if not self._validate_config():
                raise ValueError("Configuration validation failed")
            
            # Setup component-specific resources
            self._setup_resources()
            
            # Register event handlers
            self._register_event_handlers()
            
            # Component-specific initialization
            if not self._component_initialize():
                raise RuntimeError("Component-specific initialization failed")
            
            self.logger.info(f"Component {self.name} initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize component {self.name}: {e}")
            with self._state_lock:
                self._state = ComponentState.ERROR
            return False
    
    def start(self) -> None:
        """Start component operations"""
        with self._state_lock:
            if self._state != ComponentState.INITIALIZED:
                raise RuntimeError(f"Cannot start component {self.name} in state {self._state}")
            
            self._state = ComponentState.STARTING
        
        try:
            self.logger.info(f"Starting component {self.name}")
            
            # Start component-specific operations
            self._component_start()
            
            with self._state_lock:
                self._state = ComponentState.RUNNING
            
            self.logger.info(f"Component {self.name} started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start component {self.name}: {e}")
            with self._state_lock:
                self._state = ComponentState.ERROR
            raise
    
    def stop(self) -> None:
        """Stop component operations gracefully"""
        with self._state_lock:
            if self._state not in [ComponentState.RUNNING, ComponentState.ERROR]:
                self.logger.warning(f"Component {self.name} not running, current state: {self._state}")
                return
            
            self._state = ComponentState.STOPPING
        
        try:
            self.logger.info(f"Stopping component {self.name}")
            
            # Unregister event handlers
            self._unregister_event_handlers()
            
            # Component-specific cleanup
            self._component_stop()
            
            # Save state before shutdown
            self.save_state()
            
            with self._state_lock:
                self._state = ComponentState.STOPPED
            
            self.logger.info(f"Component {self.name} stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error stopping component {self.name}: {e}")
            with self._state_lock:
                self._state = ComponentState.ERROR
    
    def get_health_status(self) -> Dict[str, Any]:
        """Return comprehensive health status"""
        status = {
            'name': self.name,
            'state': self._state.value,
            'uptime': time.time() - self.metrics.start_time,
            'last_update': self.metrics.last_update_time,
            'processed_count': self.metrics.processed_count,
            'error_count': self.metrics.error_count,
            'error_rate': self._calculate_error_rate(),
            'checks': {}
        }
        
        # Run health checks
        for check_name, check_func in self._health_checks.items():
            try:
                status['checks'][check_name] = check_func()
            except Exception as e:
                status['checks'][check_name] = {'status': 'failed', 'error': str(e)}
        
        return status
    
    def get_metrics(self) -> Dict[str, float]:
        """Return component performance metrics"""
        return {
            'uptime': time.time() - self.metrics.start_time,
            'processed_count': self.metrics.processed_count,
            'error_count': self.metrics.error_count,
            'error_rate': self._calculate_error_rate(),
            'average_processing_time': self.metrics.average_processing_time,
            'memory_usage': self.metrics.memory_usage,
            'cpu_usage': self.metrics.cpu_usage
        }
    
    # Abstract methods for component-specific implementation
    @abstractmethod
    def _component_initialize(self) -> bool:
        """Component-specific initialization logic"""
        pass
    
    @abstractmethod
    def _component_start(self) -> None:
        """Component-specific start logic"""
        pass
    
    @abstractmethod
    def _component_stop(self) -> None:
        """Component-specific stop logic"""
        pass
    
    @abstractmethod
    def save_state(self) -> None:
        """Save component state"""
        pass
    
    @abstractmethod
    def load_state(self) -> None:
        """Load component state"""
        pass
    
    # Helper methods
    def _validate_config(self) -> bool:
        """Validate component configuration"""
        required_fields = self._get_required_config_fields()
        for field in required_fields:
            if field not in self.config:
                self.logger.error(f"Required configuration field missing: {field}")
                return False
        return True
    
    def _setup_resources(self) -> None:
        """Setup component resources"""
        # Default implementation - override in subclasses
        pass
    
    def _register_event_handlers(self) -> None:
        """Register event handlers with event bus"""
        for event_type, handler in self._subscriptions:
            self.event_bus.subscribe(event_type, handler)
    
    def _unregister_event_handlers(self) -> None:
        """Unregister event handlers"""
        for event_type, handler in self._subscriptions:
            self.event_bus.unsubscribe(event_type, handler)
    
    def _register_default_health_checks(self) -> None:
        """Register default health checks"""
        self._health_checks['state'] = lambda: {'status': 'healthy' if self._state == ComponentState.RUNNING else 'unhealthy'}
        self._health_checks['error_rate'] = lambda: {'status': 'healthy' if self._calculate_error_rate() < 0.05 else 'unhealthy'}
    
    def _calculate_error_rate(self) -> float:
        """Calculate error rate"""
        if self.metrics.processed_count == 0:
            return 0.0
        return self.metrics.error_count / self.metrics.processed_count
    
    def _update_metrics(self, processing_time: float = 0.0, error: bool = False) -> None:
        """Update component metrics"""
        self.metrics.last_update_time = time.time()
        self.metrics.processed_count += 1
        
        if error:
            self.metrics.error_count += 1
        
        if processing_time > 0:
            # Update average processing time (exponential moving average)
            alpha = 0.1
            self.metrics.average_processing_time = (
                alpha * processing_time + 
                (1 - alpha) * self.metrics.average_processing_time
            )
```

## Core Components

### AlgoSpace Kernel

```python
class AlgoSpaceKernel(ComponentBase):
    """Central system orchestrator and component manager"""
    
    def __init__(self, config_path: str):
        config = load_config(config_path)
        event_bus = EventBus()
        super().__init__("AlgoSpaceKernel", config, event_bus)
        
        self.components: Dict[str, ComponentBase] = {}
        self.component_dependencies: Dict[str, List[str]] = {}
        self.initialization_order: List[str] = []
    
    def _component_initialize(self) -> bool:
        """Initialize all system components"""
        try:
            # Calculate component initialization order
            self._calculate_initialization_order()
            
            # Initialize components in dependency order
            for component_name in self.initialization_order:
                self._initialize_component(component_name)
            
            return True
        except Exception as e:
            self.logger.error(f"Kernel initialization failed: {e}")
            return False
    
    def _component_start(self) -> None:
        """Start all components"""
        for component_name in self.initialization_order:
            try:
                component = self.components[component_name]
                component.start()
            except Exception as e:
                self.logger.error(f"Failed to start component {component_name}: {e}")
                raise
    
    def _component_stop(self) -> None:
        """Stop all components in reverse order"""
        for component_name in reversed(self.initialization_order):
            try:
                component = self.components[component_name]
                component.stop()
            except Exception as e:
                self.logger.error(f"Error stopping component {component_name}: {e}")
    
    def _calculate_initialization_order(self) -> None:
        """Calculate component initialization order using topological sort"""
        # Topological sort implementation
        visited = set()
        temp_visited = set()
        order = []
        
        def visit(component_name: str):
            if component_name in temp_visited:
                raise ValueError(f"Circular dependency detected involving {component_name}")
            
            if component_name not in visited:
                temp_visited.add(component_name)
                
                # Visit dependencies first
                for dependency in self.component_dependencies.get(component_name, []):
                    visit(dependency)
                
                temp_visited.remove(component_name)
                visited.add(component_name)
                order.append(component_name)
        
        for component_name in self.components.keys():
            if component_name not in visited:
                visit(component_name)
        
        self.initialization_order = order
    
    def get_component(self, name: str) -> Optional[ComponentBase]:
        """Get component by name"""
        return self.components.get(name)
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status"""
        health_status = {
            'kernel': self.get_health_status(),
            'components': {},
            'overall_status': 'healthy'
        }
        
        unhealthy_components = 0
        for name, component in self.components.items():
            component_health = component.get_health_status()
            health_status['components'][name] = component_health
            
            if component_health['state'] != 'running':
                unhealthy_components += 1
        
        # Determine overall status
        if unhealthy_components == 0:
            health_status['overall_status'] = 'healthy'
        elif unhealthy_components < len(self.components) * 0.5:
            health_status['overall_status'] = 'degraded'
        else:
            health_status['overall_status'] = 'unhealthy'
        
        return health_status
```

### Event Bus Component

```python
class EventBusComponent(ComponentBase):
    """Enhanced event bus with monitoring and performance optimization"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("EventBus", config, None)  # EventBus doesn't need itself
        
        self._subscribers: Dict[EventType, List[Callable]] = {}
        self._event_queue = asyncio.Queue(maxsize=config.get('queue_size', 10000))
        self._worker_tasks: List[asyncio.Task] = []
        self._event_stats: Dict[EventType, Dict[str, int]] = {}
        self._performance_monitor = EventPerformanceMonitor()
    
    def _component_initialize(self) -> bool:
        """Initialize event bus"""
        try:
            # Initialize event statistics
            for event_type in EventType:
                self._event_stats[event_type] = {
                    'published': 0,
                    'processed': 0,
                    'failed': 0
                }
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize EventBus: {e}")
            return False
    
    def _component_start(self) -> None:
        """Start event processing workers"""
        num_workers = self.config.get('worker_threads', 4)
        
        for i in range(num_workers):
            task = asyncio.create_task(self._event_worker(f"worker-{i}"))
            self._worker_tasks.append(task)
        
        self.logger.info(f"Started {num_workers} event processing workers")
    
    def _component_stop(self) -> None:
        """Stop event processing workers"""
        # Cancel all worker tasks
        for task in self._worker_tasks:
            task.cancel()
        
        # Wait for all tasks to complete
        asyncio.gather(*self._worker_tasks, return_exceptions=True)
        
        self._worker_tasks.clear()
    
    async def _event_worker(self, worker_name: str) -> None:
        """Event processing worker"""
        self.logger.debug(f"Event worker {worker_name} started")
        
        try:
            while True:
                # Get event from queue
                event = await self._event_queue.get()
                
                # Process event
                await self._process_event(event)
                
                # Mark task as done
                self._event_queue.task_done()
                
        except asyncio.CancelledError:
            self.logger.debug(f"Event worker {worker_name} cancelled")
        except Exception as e:
            self.logger.error(f"Event worker {worker_name} error: {e}")
    
    async def _process_event(self, event: Event) -> None:
        """Process single event with error handling"""
        start_time = time.time()
        
        try:
            subscribers = self._subscribers.get(event.event_type, [])
            
            # Process all subscribers
            for subscriber in subscribers:
                try:
                    if asyncio.iscoroutinefunction(subscriber):
                        await subscriber(event)
                    else:
                        subscriber(event)
                except Exception as e:
                    self.logger.error(f"Subscriber error for {event.event_type}: {e}")
                    self._event_stats[event.event_type]['failed'] += 1
            
            # Update statistics
            self._event_stats[event.event_type]['processed'] += 1
            
            # Record performance metrics
            processing_time = time.time() - start_time
            self._performance_monitor.record_event_processing(event.event_type, processing_time)
            
        except Exception as e:
            self.logger.error(f"Event processing error: {e}")
            self._event_stats[event.event_type]['failed'] += 1
    
    def subscribe(self, event_type: EventType, callback: Callable[[Event], None]) -> None:
        """Subscribe to event type"""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        
        self._subscribers[event_type].append(callback)
        self.logger.debug(f"Subscriber registered for {event_type}")
    
    def unsubscribe(self, event_type: EventType, callback: Callable[[Event], None]) -> None:
        """Unsubscribe from event type"""
        if event_type in self._subscribers:
            try:
                self._subscribers[event_type].remove(callback)
                self.logger.debug(f"Subscriber removed for {event_type}")
            except ValueError:
                self.logger.warning(f"Callback not found for {event_type}")
    
    async def publish(self, event: Event) -> None:
        """Publish event to queue"""
        try:
            await self._event_queue.put(event)
            self._event_stats[event.event_type]['published'] += 1
            
        except asyncio.QueueFull:
            self.logger.error(f"Event queue full, dropping event: {event.event_type}")
            self._event_stats[event.event_type]['failed'] += 1
    
    def get_event_statistics(self) -> Dict[str, Any]:
        """Get event processing statistics"""
        return {
            'queue_size': self._event_queue.qsize(),
            'active_workers': len([t for t in self._worker_tasks if not t.done()]),
            'event_stats': self._event_stats,
            'performance_metrics': self._performance_monitor.get_metrics()
        }
```

## Data Pipeline Components

### Data Handler Component

```python
class DataHandlerComponent(ComponentBase):
    """Handle market data ingestion and validation"""
    
    def __init__(self, name: str, config: Dict[str, Any], event_bus: EventBus):
        super().__init__(name, config, event_bus)
        
        self.data_source = None
        self.validator = DataValidator(config.get('validation', {}))
        self.normalizer = DataNormalizer(config.get('normalization', {}))
        self.connection_manager = ConnectionManager(config.get('connection', {}))
        
        # Subscribe to system events
        self._subscriptions = [
            (EventType.SYSTEM_SHUTDOWN, self._handle_shutdown)
        ]
    
    def _component_initialize(self) -> bool:
        """Initialize data handler"""
        try:
            # Initialize data source
            source_type = self.config['type']
            self.data_source = self._create_data_source(source_type)
            
            # Setup connection monitoring
            self.connection_manager.setup_monitoring()
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize data handler: {e}")
            return False
    
    def _component_start(self) -> None:
        """Start data stream"""
        try:
            # Connect to data source
            if not self.data_source.connect():
                raise RuntimeError("Failed to connect to data source")
            
            # Subscribe to symbols
            symbols = self.config.get('symbols', [])
            self.data_source.subscribe(symbols)
            
            # Start connection monitoring
            asyncio.create_task(self.connection_manager.maintain_connections())
            
            self.logger.info(f"Data handler started for symbols: {symbols}")
            
        except Exception as e:
            self.logger.error(f"Failed to start data handler: {e}")
            raise
    
    def _component_stop(self) -> None:
        """Stop data stream"""
        try:
            if self.data_source:
                self.data_source.disconnect()
            
            self.connection_manager.close_all_connections()
            
        except Exception as e:
            self.logger.error(f"Error stopping data handler: {e}")
    
    def on_tick_received(self, tick_data: TickData) -> None:
        """Handle incoming tick data"""
        start_time = time.time()
        
        try:
            # Validate tick data
            is_valid, errors = self.validator.validate_tick(tick_data)
            if not is_valid:
                self.logger.warning(f"Invalid tick data: {errors}")
                self._update_metrics(time.time() - start_time, error=True)
                return
            
            # Normalize tick data
            normalized_tick = self.normalizer.normalize_tick(tick_data)
            
            # Publish tick event
            event = Event(
                event_type=EventType.NEW_TICK,
                timestamp=datetime.now(),
                payload=normalized_tick,
                source=self.name
            )
            
            asyncio.create_task(self.event_bus.publish(event))
            
            # Update metrics
            self._update_metrics(time.time() - start_time)
            
        except Exception as e:
            self.logger.error(f"Error processing tick: {e}")
            self._update_metrics(time.time() - start_time, error=True)
    
    def _create_data_source(self, source_type: str) -> MarketDataSource:
        """Factory method for creating data sources"""
        if source_type == 'rithmic':
            return RithmicDataSource(self.config['connection'], self.on_tick_received)
        elif source_type == 'ib':
            return InteractiveBrokersDataSource(self.config['connection'], self.on_tick_received)
        elif source_type == 'backtest':
            return BacktestDataSource(self.config['data_file'], self.on_tick_received)
        else:
            raise ValueError(f"Unknown data source type: {source_type}")
    
    def _handle_shutdown(self, event: Event) -> None:
        """Handle system shutdown"""
        self.logger.info("Received shutdown signal")
        self.stop()
    
    def save_state(self) -> None:
        """Save data handler state"""
        state = {
            'last_tick_time': self.metrics.last_update_time,
            'processed_count': self.metrics.processed_count,
            'error_count': self.metrics.error_count
        }
        
        # Save to file or database
        with open(f"state/{self.name}_state.json", 'w') as f:
            json.dump(state, f)
    
    def load_state(self) -> None:
        """Load data handler state"""
        try:
            with open(f"state/{self.name}_state.json", 'r') as f:
                state = json.load(f)
                
            # Restore metrics
            self.metrics.last_update_time = state.get('last_tick_time', time.time())
            self.metrics.processed_count = state.get('processed_count', 0)
            self.metrics.error_count = state.get('error_count', 0)
            
        except FileNotFoundError:
            self.logger.info("No saved state found")
    
    def _get_required_config_fields(self) -> List[str]:
        """Get required configuration fields"""
        return ['type', 'symbols']
```

### Bar Generator Component

```python
class BarGeneratorComponent(ComponentBase):
    """Generate OHLCV bars from tick data"""
    
    def __init__(self, name: str, config: Dict[str, Any], event_bus: EventBus):
        super().__init__(name, config, event_bus)
        
        self.timeframes = config.get('timeframes', [5, 30])
        self.current_bars: Dict[Tuple[str, int], BarData] = {}
        self.bar_buffers: Dict[Tuple[str, int], List[BarData]] = {}
        self.tick_buffer: List[TickData] = []
        self.buffer_size = config.get('buffer_size', 1000)
        
        # Subscribe to tick events
        self._subscriptions = [
            (EventType.NEW_TICK, self.on_new_tick)
        ]
    
    def _component_initialize(self) -> bool:
        """Initialize bar generator"""
        try:
            # Initialize bar buffers for each symbol and timeframe
            symbols = self.config.get('symbols', [])
            for symbol in symbols:
                for timeframe in self.timeframes:
                    key = (symbol, timeframe)
                    self.bar_buffers[key] = []
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize bar generator: {e}")
            return False
    
    def on_new_tick(self, event: Event) -> None:
        """Process new tick data"""
        start_time = time.time()
        
        try:
            tick_data = event.payload
            
            # Add to tick buffer for batch processing
            self.tick_buffer.append(tick_data)
            
            # Process buffer if it's full
            if len(self.tick_buffer) >= self.buffer_size:
                self._process_tick_buffer()
            
            # Update bars for each timeframe
            for timeframe in self.timeframes:
                self._update_bar(tick_data, timeframe)
            
            self._update_metrics(time.time() - start_time)
            
        except Exception as e:
            self.logger.error(f"Error processing tick in bar generator: {e}")
            self._update_metrics(time.time() - start_time, error=True)
    
    def _process_tick_buffer(self) -> None:
        """Process accumulated tick buffer"""
        if not self.tick_buffer:
            return
        
        # Batch processing optimizations
        symbols = set(tick.symbol for tick in self.tick_buffer)
        
        for symbol in symbols:
            symbol_ticks = [tick for tick in self.tick_buffer if tick.symbol == symbol]
            self._batch_process_symbol_ticks(symbol, symbol_ticks)
        
        # Clear buffer
        self.tick_buffer.clear()
    
    def _batch_process_symbol_ticks(self, symbol: str, ticks: List[TickData]) -> None:
        """Batch process ticks for a specific symbol"""
        # Vectorized calculations
        prices = np.array([tick.price for tick in ticks])
        volumes = np.array([tick.volume for tick in ticks])
        
        # Calculate batch statistics
        vwap = np.sum(prices * volumes) / np.sum(volumes) if np.sum(volumes) > 0 else 0
        price_volatility = np.std(prices) if len(prices) > 1 else 0
        volume_imbalance = (volumes.max() - volumes.min()) / volumes.mean() if volumes.mean() > 0 else 0
        
        # Store batch analytics
        batch_analytics = {
            'symbol': symbol,
            'tick_count': len(ticks),
            'vwap': vwap,
            'volatility': price_volatility,
            'volume_imbalance': volume_imbalance,
            'time_span': ticks[-1].timestamp - ticks[0].timestamp
        }
        
        # Optionally publish batch analytics
        self._publish_batch_analytics(batch_analytics)
    
    def _update_bar(self, tick: TickData, timeframe: int) -> None:
        """Update bar for specific timeframe"""
        bar_key = (tick.symbol, timeframe)
        bar_timestamp = self._align_to_timeframe(tick.timestamp, timeframe)
        
        # Get or create current bar
        if bar_key not in self.current_bars:
            self.current_bars[bar_key] = self._create_new_bar(tick, bar_timestamp, timeframe)
        
        current_bar = self.current_bars[bar_key]
        
        # Check if we need to close current bar
        if bar_timestamp > current_bar.timestamp:
            self._close_bar(current_bar)
            self.current_bars[bar_key] = self._create_new_bar(tick, bar_timestamp, timeframe)
            current_bar = self.current_bars[bar_key]
        
        # Update current bar
        self._update_bar_with_tick(current_bar, tick)
    
    def _close_bar(self, bar: BarData) -> None:
        """Close completed bar and publish event"""
        try:
            # Store in buffer
            bar_key = (bar.symbol, bar.timeframe)
            self.bar_buffers[bar_key].append(bar)
            
            # Maintain buffer size
            max_buffer_size = 1000
            if len(self.bar_buffers[bar_key]) > max_buffer_size:
                self.bar_buffers[bar_key] = self.bar_buffers[bar_key][-max_buffer_size:]
            
            # Publish bar event
            event_type = self._get_bar_event_type(bar.timeframe)
            event = Event(
                event_type=event_type,
                timestamp=datetime.now(),
                payload=bar,
                source=self.name
            )
            
            asyncio.create_task(self.event_bus.publish(event))
            
            self.logger.debug(f"Closed {bar.timeframe}min bar for {bar.symbol}: "
                            f"O={bar.open}, H={bar.high}, L={bar.low}, C={bar.close}, V={bar.volume}")
            
        except Exception as e:
            self.logger.error(f"Error closing bar: {e}")
    
    def _get_bar_event_type(self, timeframe: int) -> EventType:
        """Get appropriate event type for timeframe"""
        if timeframe == 5:
            return EventType.NEW_5MIN_BAR
        elif timeframe == 30:
            return EventType.NEW_30MIN_BAR
        else:
            return EventType.NEW_BAR
    
    def get_bar_history(self, symbol: str, timeframe: int, count: int = 100) -> List[BarData]:
        """Get recent bar history for symbol and timeframe"""
        bar_key = (symbol, timeframe)
        bars = self.bar_buffers.get(bar_key, [])
        return bars[-count:] if bars else []
```

## Analysis Components

### Indicator Engine Component

```python
class IndicatorEngineComponent(ComponentBase):
    """Calculate technical indicators from bar data"""
    
    def __init__(self, name: str, config: Dict[str, Any], event_bus: EventBus):
        super().__init__(name, config, event_bus)
        
        self.indicators = {}
        self.calculation_cache = {}
        self.price_buffers: Dict[Tuple[str, int], List[BarData]] = {}
        self.indicator_calculators = self._initialize_calculators()
        
        # Subscribe to bar events
        self._subscriptions = [
            (EventType.NEW_5MIN_BAR, self.on_new_bar),
            (EventType.NEW_30MIN_BAR, self.on_new_bar)
        ]
    
    def _component_initialize(self) -> bool:
        """Initialize indicator engine"""
        try:
            # Initialize indicator calculators
            enabled_indicators = self.config.get('enabled_indicators', [])
            
            for indicator_name in enabled_indicators:
                calculator_class = self._get_calculator_class(indicator_name)
                calculator_config = self.config.get(indicator_name, {})
                self.indicator_calculators[indicator_name] = calculator_class(calculator_config)
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize indicator engine: {e}")
            return False
    
    def on_new_bar(self, event: Event) -> None:
        """Process new bar and calculate indicators"""
        start_time = time.time()
        
        try:
            bar_data = event.payload
            
            # Update price buffer
            self._update_price_buffer(bar_data)
            
            # Calculate indicators
            indicator_results = self._calculate_all_indicators(bar_data)
            
            # Publish indicator results
            if indicator_results:
                self._publish_indicator_results(bar_data, indicator_results)
            
            self._update_metrics(time.time() - start_time)
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")
            self._update_metrics(time.time() - start_time, error=True)
    
    def _calculate_all_indicators(self, bar: BarData) -> Dict[str, Any]:
        """Calculate all indicators for the given bar"""
        results = {}
        
        # Get price history
        price_history = self._get_price_history(bar.symbol, bar.timeframe)
        
        if len(price_history) < 50:  # Minimum history required
            return results
        
        # Calculate each indicator
        for indicator_name, calculator in self.indicator_calculators.items():
            try:
                # Check cache first
                cache_key = self._get_cache_key(indicator_name, bar.symbol, bar.timestamp)
                
                if cache_key in self.calculation_cache:
                    indicator_result = self.calculation_cache[cache_key]
                else:
                    # Calculate indicator
                    indicator_result = calculator.calculate(price_history)
                    
                    # Cache result
                    self.calculation_cache[cache_key] = indicator_result
                    
                    # Maintain cache size
                    if len(self.calculation_cache) > 10000:
                        self._cleanup_cache()
                
                results.update(indicator_result)
                
            except Exception as e:
                self.logger.error(f"Error calculating {indicator_name}: {e}")
        
        return results
    
    def _initialize_calculators(self) -> Dict[str, Any]:
        """Initialize indicator calculators"""
        return {
            'mlmi': MLMICalculator(),
            'nwrqk': NWRQKCalculator(),
            'fvg': FVGCalculator(),
            'lvn': LVNCalculator(),
            'time_features': TimeFeaturesCalculator()
        }
    
    def _publish_indicator_results(self, bar: BarData, results: Dict[str, Any]) -> None:
        """Publish calculated indicator results"""
        payload = {
            'symbol': bar.symbol,
            'timeframe': bar.timeframe,
            'timestamp': bar.timestamp,
            'indicators': results
        }
        
        event = Event(
            event_type=EventType.INDICATORS_READY,
            timestamp=datetime.now(),
            payload=payload,
            source=self.name
        )
        
        asyncio.create_task(self.event_bus.publish(event))
```

## Intelligence Components

### MARL Component

```python
class MARLComponent(ComponentBase):
    """Multi-Agent Reinforcement Learning component"""
    
    def __init__(self, name: str, config: Dict[str, Any], event_bus: EventBus):
        super().__init__(name, config, event_bus)
        
        self.agents: Dict[str, Any] = {}
        self.environment = None
        self.coordination_mechanism = None
        self.decision_cache: Dict[str, Any] = {}
        self.performance_tracker = MARLPerformanceTracker()
        
        # Subscribe to events
        self._subscriptions = [
            (EventType.SYNERGY_DETECTED, self.on_synergy_detected),
            (EventType.NEW_30MIN_BAR, self.on_market_update),
            (EventType.PORTFOLIO_UPDATE, self.on_portfolio_update)
        ]
    
    def _component_initialize(self) -> bool:
        """Initialize MARL system"""
        try:
            # Initialize agents
            self._initialize_agents()
            
            # Initialize environment
            self._initialize_environment()
            
            # Initialize coordination mechanism
            self._initialize_coordination()
            
            # Load pre-trained models
            self._load_models()
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize MARL component: {e}")
            return False
    
    def _initialize_agents(self) -> None:
        """Initialize MARL agents"""
        agent_configs = self.config.get('agents', {})
        
        for agent_name, agent_config in agent_configs.items():
            agent_class = self._get_agent_class(agent_name)
            self.agents[agent_name] = agent_class(agent_config)
            
        self.logger.info(f"Initialized {len(self.agents)} MARL agents")
    
    def on_synergy_detected(self, event: Event) -> None:
        """Handle synergy detection events"""
        start_time = time.time()
        
        try:
            synergy_data = event.payload
            
            # Process with MARL agents
            decision = self._make_coordinated_decision(synergy_data)
            
            if decision and decision.get('action') != 'hold':
                # Publish trading decision
                self._publish_trading_decision(decision)
            
            self._update_metrics(time.time() - start_time)
            
        except Exception as e:
            self.logger.error(f"Error processing synergy detection: {e}")
            self._update_metrics(time.time() - start_time, error=True)
    
    def _make_coordinated_decision(self, market_state: Dict) -> Optional[Dict]:
        """Make coordinated decision using all agents"""
        # Prepare observations for each agent
        observations = self._prepare_agent_observations(market_state)
        
        # Get individual agent decisions
        agent_decisions = {}
        for agent_name, agent in self.agents.items():
            try:
                observation = observations.get(agent_name)
                if observation is not None:
                    decision = agent.select_action(observation, training=False)
                    agent_decisions[agent_name] = decision
            except Exception as e:
                self.logger.error(f"Error getting decision from {agent_name}: {e}")
        
        # Coordinate decisions
        coordinated_decision = self.coordination_mechanism.coordinate(agent_decisions, market_state)
        
        # Track performance
        self.performance_tracker.record_decision(coordinated_decision, market_state)
        
        return coordinated_decision
    
    def _get_agent_class(self, agent_name: str) -> type:
        """Get agent class by name"""
        agent_classes = {
            'strategic': StrategicAgent,
            'tactical': TacticalAgent,
            'risk': RiskAgent
        }
        return agent_classes.get(agent_name, StrategicAgent)
```

## Component Lifecycle Management

### Component Manager

```python
class ComponentManager:
    """Manage component lifecycle and dependencies"""
    
    def __init__(self, kernel: AlgoSpaceKernel):
        self.kernel = kernel
        self.components: Dict[str, ComponentBase] = {}
        self.dependency_graph = DependencyGraph()
        self.health_monitor = ComponentHealthMonitor()
        
    def register_component(self, component: ComponentBase, dependencies: List[str] = None) -> None:
        """Register component with dependency information"""
        self.components[component.name] = component
        
        if dependencies:
            for dependency in dependencies:
                self.dependency_graph.add_dependency(component.name, dependency)
    
    def initialize_all_components(self) -> bool:
        """Initialize all components in dependency order"""
        try:
            # Get initialization order
            init_order = self.dependency_graph.get_topological_order()
            
            # Initialize components
            for component_name in init_order:
                component = self.components[component_name]
                
                if not component.initialize():
                    raise RuntimeError(f"Failed to initialize {component_name}")
                
                self.logger.info(f"Initialized component: {component_name}")
            
            return True
        except Exception as e:
            self.logger.error(f"Component initialization failed: {e}")
            return False
    
    def start_all_components(self) -> None:
        """Start all components"""
        init_order = self.dependency_graph.get_topological_order()
        
        for component_name in init_order:
            component = self.components[component_name]
            component.start()
            self.logger.info(f"Started component: {component_name}")
    
    def stop_all_components(self) -> None:
        """Stop all components in reverse order"""
        init_order = self.dependency_graph.get_topological_order()
        
        for component_name in reversed(init_order):
            try:
                component = self.components[component_name]
                component.stop()
                self.logger.info(f"Stopped component: {component_name}")
            except Exception as e:
                self.logger.error(f"Error stopping {component_name}: {e}")
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health"""
        health_data = {}
        
        for name, component in self.components.items():
            health_data[name] = component.get_health_status()
        
        return {
            'components': health_data,
            'overall_health': self._calculate_overall_health(health_data)
        }
    
    def _calculate_overall_health(self, health_data: Dict[str, Any]) -> str:
        """Calculate overall system health"""
        healthy_count = sum(1 for health in health_data.values() 
                          if health.get('state') == 'running')
        
        total_count = len(health_data)
        health_ratio = healthy_count / total_count if total_count > 0 else 0
        
        if health_ratio >= 0.9:
            return 'healthy'
        elif health_ratio >= 0.7:
            return 'degraded'
        else:
            return 'unhealthy'
```

## Inter-Component Communication

### Message Patterns

```python
class ComponentCommunicationPatterns:
    """Standard communication patterns between components"""
    
    @staticmethod
    def request_response_pattern(sender: ComponentBase, 
                               receiver: ComponentBase, 
                               request_data: Dict) -> Dict:
        """Synchronous request-response pattern"""
        
        # Create request event
        request_event = Event(
            event_type=EventType.COMPONENT_REQUEST,
            timestamp=datetime.now(),
            payload={
                'sender': sender.name,
                'receiver': receiver.name,
                'request_id': str(uuid.uuid4()),
                'data': request_data
            },
            source=sender.name
        )
        
        # Setup response handler
        response_future = asyncio.Future()
        
        def response_handler(event: Event):
            if (event.payload.get('request_id') == request_event.payload['request_id'] and
                event.payload.get('receiver') == sender.name):
                response_future.set_result(event.payload['data'])
        
        # Subscribe to response
        receiver.event_bus.subscribe(EventType.COMPONENT_RESPONSE, response_handler)
        
        try:
            # Send request
            asyncio.create_task(receiver.event_bus.publish(request_event))
            
            # Wait for response with timeout
            response = asyncio.wait_for(response_future, timeout=5.0)
            return response
            
        finally:
            # Cleanup
            receiver.event_bus.unsubscribe(EventType.COMPONENT_RESPONSE, response_handler)
    
    @staticmethod
    def pub_sub_pattern(publisher: ComponentBase, 
                       event_type: EventType, 
                       data: Any) -> None:
        """Asynchronous publish-subscribe pattern"""
        
        event = Event(
            event_type=event_type,
            timestamp=datetime.now(),
            payload=data,
            source=publisher.name
        )
        
        asyncio.create_task(publisher.event_bus.publish(event))
    
    @staticmethod
    def pipeline_pattern(components: List[ComponentBase], 
                        initial_data: Any) -> None:
        """Sequential processing pipeline pattern"""
        
        current_data = initial_data
        
        for i, component in enumerate(components):
            # Create pipeline event
            event = Event(
                event_type=EventType.PIPELINE_STAGE,
                timestamp=datetime.now(),
                payload={
                    'stage': i,
                    'data': current_data,
                    'next_component': components[i + 1].name if i + 1 < len(components) else None
                },
                source=component.name
            )
            
            # Process in component
            result = component.process_pipeline_stage(event)
            current_data = result
```

## Component Testing Strategy

### Component Test Base

```python
class ComponentTestBase(unittest.TestCase):
    """Base class for component testing"""
    
    def setUp(self):
        """Setup test environment"""
        self.test_config = self._get_test_config()
        self.mock_event_bus = MagicMock(spec=EventBus)
        self.component = None
    
    def tearDown(self):
        """Cleanup test environment"""
        if self.component and hasattr(self.component, 'stop'):
            self.component.stop()
    
    def _get_test_config(self) -> Dict[str, Any]:
        """Get test configuration"""
        return {
            'test_mode': True,
            'log_level': 'DEBUG',
            'timeout': 1.0
        }
    
    def test_component_initialization(self):
        """Test component initialization"""
        self.assertTrue(self.component.initialize())
        self.assertEqual(self.component._state, ComponentState.INITIALIZED)
    
    def test_component_start_stop(self):
        """Test component start/stop lifecycle"""
        self.component.initialize()
        self.component.start()
        self.assertEqual(self.component._state, ComponentState.RUNNING)
        
        self.component.stop()
        self.assertEqual(self.component._state, ComponentState.STOPPED)
    
    def test_component_health_check(self):
        """Test component health reporting"""
        self.component.initialize()
        self.component.start()
        
        health = self.component.get_health_status()
        self.assertIn('name', health)
        self.assertIn('state', health)
        self.assertIn('checks', health)
    
    def test_component_metrics(self):
        """Test component metrics collection"""
        metrics = self.component.get_metrics()
        self.assertIn('uptime', metrics)
        self.assertIn('processed_count', metrics)
        self.assertIn('error_count', metrics)
    
    def test_event_handling(self):
        """Test event subscription and handling"""
        test_event = Event(
            event_type=EventType.NEW_TICK,
            timestamp=datetime.now(),
            payload={'test': 'data'},
            source='test'
        )
        
        # Test event handling
        self.component.initialize()
        self.component.start()
        
        # Simulate event
        if hasattr(self.component, 'on_new_tick'):
            self.component.on_new_tick(test_event)
            
        # Verify processing
        self.assertGreater(self.component.metrics.processed_count, 0)
```

This component design architecture provides a robust foundation for building scalable, maintainable, and testable trading system components. Each component follows consistent patterns while maintaining the flexibility to implement specific functionality.

## Related Documentation

- [System Overview](system_overview.md)
- [Data Flow Architecture](data_flow.md)
- [API Documentation](../api/)
- [Development Guidelines](../development/coding_standards.md)
- [Testing Guide](../development/testing_guidelines.md)