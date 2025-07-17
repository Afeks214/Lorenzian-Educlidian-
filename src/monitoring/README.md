# System Monitoring and Health Management

## Overview

The monitoring system provides comprehensive real-time observability for the GrandModel trading system. It includes health monitoring, performance metrics collection, alerting, and automated recovery mechanisms to ensure system reliability and optimal performance in production environments.

## Core Components

### Health Monitor (`health_monitor.py`)

Central health monitoring system that tracks the status of all system components.

**Key Features:**
- Real-time component health tracking
- Automated health checks with configurable intervals
- Health state transitions with event notifications
- Dependency-aware health propagation
- Performance degradation detection

**Usage:**
```python
from src.monitoring.health_monitor import HealthMonitor

# Initialize health monitor
health_monitor = HealthMonitor(config, event_bus)

# Register components for monitoring
health_monitor.register_component('strategic_agent', {
    'check_interval': 30,  # seconds
    'timeout': 5,
    'critical': True
})

# Start monitoring
health_monitor.start()

# Check system health
overall_health = health_monitor.get_system_health()
print(f"System Status: {overall_health.status}")
print(f"Healthy Components: {overall_health.healthy_count}/{overall_health.total_count}")
```

**Health States:**
- `HEALTHY`: Component operating normally
- `WARNING`: Performance degradation detected
- `CRITICAL`: Component errors but still functional
- `FAILED`: Component non-functional
- `UNKNOWN`: Health status cannot be determined

### Service Health State Machine (`service_health_state_machine.py`)

Advanced state machine for managing component health transitions.

**State Transitions:**
```python
class ServiceHealthStateMachine:
    """Manages health state transitions with hysteresis"""
    
    def __init__(self, config):
        self.current_state = HealthState.UNKNOWN
        self.state_history = deque(maxlen=100)
        self.thresholds = config['thresholds']
        
        # State transition matrix
        self.transitions = {
            HealthState.HEALTHY: {
                HealthState.WARNING: self._check_warning_conditions,
                HealthState.CRITICAL: self._check_critical_conditions
            },
            HealthState.WARNING: {
                HealthState.HEALTHY: self._check_recovery_conditions,
                HealthState.CRITICAL: self._check_degradation_conditions
            },
            HealthState.CRITICAL: {
                HealthState.WARNING: self._check_improvement_conditions,
                HealthState.FAILED: self._check_failure_conditions
            }
        }
    
    def update_health(self, metrics: Dict[str, float]) -> HealthState:
        """Update health state based on current metrics"""
        new_state = self._evaluate_health_state(metrics)
        
        if new_state != self.current_state:
            if self._should_transition(self.current_state, new_state, metrics):
                old_state = self.current_state
                self.current_state = new_state
                self._record_transition(old_state, new_state, metrics)
                
                # Publish state change event
                self._publish_state_change_event(old_state, new_state)
        
        return self.current_state
    
    def _should_transition(self, from_state, to_state, metrics):
        """Apply hysteresis to prevent state flapping"""
        transition_func = self.transitions.get(from_state, {}).get(to_state)
        
        if transition_func:
            return transition_func(metrics)
        
        return False
```

### Metrics Exporter (`metrics_exporter.py`)

High-performance metrics collection and export system.

**Features:**
- Real-time metrics aggregation
- Multiple export formats (Prometheus, InfluxDB, JSON)
- Configurable retention policies
- Batch processing for efficiency
- Memory-efficient circular buffers

**Usage:**
```python
from src.monitoring.metrics_exporter import MetricsExporter

# Initialize exporter
exporter = MetricsExporter(config={
    'export_interval': 60,  # seconds
    'batch_size': 1000,
    'retention_hours': 24,
    'exporters': ['prometheus', 'influxdb']
})

# Record metrics
exporter.record_counter('orders_processed', 1, tags={'agent': 'strategic'})
exporter.record_gauge('latency_ms', 2.5, tags={'component': 'matrix_assembler'})
exporter.record_histogram('execution_time_ms', 150.0)

# Export metrics
await exporter.export_metrics()
```

**Metric Types:**
```python
# Counter metrics (always increasing)
exporter.increment_counter('api_requests_total', tags={'endpoint': '/health'})

# Gauge metrics (current value)
exporter.set_gauge('memory_usage_mb', 512.5)

# Histogram metrics (distribution)
exporter.observe_histogram('request_duration_seconds', 0.25)

# Summary metrics (quantiles)
exporter.observe_summary('response_time_ms', 45.0)
```

### Tactical Health Monitor (`tactical_health.py`)

Specialized health monitoring for high-frequency tactical components.

**Features:**
- Sub-second health checks
- Latency-sensitive monitoring
- Execution path health tracking
- Real-time alerting for critical failures

**Performance Metrics:**
```python
class TacticalHealthMonitor:
    """High-frequency health monitoring for tactical components"""
    
    def __init__(self, config):
        self.config = config
        self.check_interval_ms = config.get('check_interval_ms', 100)  # 100ms
        self.latency_thresholds = config['latency_thresholds']
        
        # Performance tracking
        self.latency_tracker = LatencyTracker(window_size=1000)
        self.error_tracker = ErrorTracker(window_size=100)
        
    async def monitor_tactical_health(self):
        """Continuous tactical health monitoring"""
        while self.monitoring_active:
            start_time = time.perf_counter()
            
            # Check tactical components
            health_status = await self._check_tactical_components()
            
            # Update latency tracking
            check_duration = (time.perf_counter() - start_time) * 1000  # ms
            self.latency_tracker.record(check_duration)
            
            # Alert on performance issues
            if check_duration > self.latency_thresholds['critical']:
                await self._send_critical_latency_alert(check_duration)
            
            # Wait for next check
            await asyncio.sleep(self.check_interval_ms / 1000.0)
    
    async def _check_tactical_components(self):
        """Check health of tactical trading components"""
        components = [
            'tactical_matrix_assembler',
            'tactical_fvg_detector',
            'order_router',
            'execution_engine'
        ]
        
        health_results = {}
        
        for component in components:
            try:
                # Perform health check with timeout
                health_check = asyncio.wait_for(
                    self._component_health_check(component),
                    timeout=0.050  # 50ms timeout
                )
                
                result = await health_check
                health_results[component] = result
                
            except asyncio.TimeoutError:
                health_results[component] = {'status': 'TIMEOUT', 'latency': float('inf')}
                self.error_tracker.record_error(component, 'timeout')
            except Exception as e:
                health_results[component] = {'status': 'ERROR', 'error': str(e)}
                self.error_tracker.record_error(component, str(e))
        
        return health_results
```

### Logger Configuration (`logger_config.py`)

Centralized logging configuration with performance optimization.

**Features:**
- Structured logging with JSON format
- Log level filtering and rotation
- Performance-optimized async logging
- Context-aware log enrichment
- Integration with monitoring systems

**Configuration:**
```python
import logging
from src.monitoring.logger_config import setup_logging

# Setup production logging
setup_logging({
    'level': 'INFO',
    'format': 'structured',
    'handlers': [
        {
            'type': 'rotating_file',
            'filename': 'logs/grandmodel.log',
            'max_bytes': 100_000_000,  # 100MB
            'backup_count': 10
        },
        {
            'type': 'console',
            'level': 'WARNING'  # Only warnings and errors to console
        },
        {
            'type': 'syslog',
            'address': ('localhost', 514),
            'facility': 'daemon'
        }
    ],
    'enrichment': {
        'add_thread_id': True,
        'add_process_id': True,
        'add_hostname': True,
        'add_component_context': True
    },
    'performance': {
        'async_logging': True,
        'buffer_size': 10000,
        'flush_interval': 1.0
    }
})

# Use structured logging
logger = logging.getLogger('grandmodel.strategic_agent')

logger.info(
    "Strategic decision made",
    extra={
        'confidence': 0.85,
        'position_size': 0.02,
        'pattern_type': 'TYPE_1',
        'execution_time_ms': 2.3
    }
)
```

## Live Performance Monitoring

### Performance Dashboard (`live_performance/`)

Real-time performance monitoring dashboard components.

**Key Metrics:**
- System throughput (events/second, trades/minute)
- Latency percentiles (P50, P95, P99)
- Memory and CPU utilization
- Component health status
- Error rates and recovery times

**Dashboard Components:**
```python
# live_performance/dashboard.py
class PerformanceDashboard:
    """Real-time performance monitoring dashboard"""
    
    def __init__(self, config, event_bus):
        self.config = config
        self.event_bus = event_bus
        self.metrics_collector = MetricsCollector()
        
        # Subscribe to performance events
        event_bus.subscribe(EventType.PERFORMANCE_METRIC, self.on_performance_metric)
        event_bus.subscribe(EventType.HEALTH_STATUS_CHANGED, self.on_health_change)
    
    async def on_performance_metric(self, event):
        """Handle incoming performance metrics"""
        metric = event.payload
        
        # Update real-time metrics
        self.metrics_collector.update_metric(
            metric['name'],
            metric['value'],
            metric.get('tags', {}),
            metric['timestamp']
        )
        
        # Check for performance alerts
        await self._check_performance_alerts(metric)
    
    def get_dashboard_data(self):
        """Get current dashboard data"""
        return {
            'system_overview': self._get_system_overview(),
            'component_health': self._get_component_health(),
            'performance_metrics': self._get_performance_metrics(),
            'recent_alerts': self._get_recent_alerts(),
            'throughput_charts': self._get_throughput_charts()
        }
    
    def _get_system_overview(self):
        """Get high-level system metrics"""
        return {
            'uptime': self._calculate_uptime(),
            'total_events_processed': self.metrics_collector.get_counter('events_processed'),
            'current_throughput': self.metrics_collector.get_rate('events_processed'),
            'avg_latency_ms': self.metrics_collector.get_average('processing_latency_ms'),
            'error_rate': self.metrics_collector.get_rate('errors')
        }
```

### Real-Time Alerting

```python
class AlertManager:
    """Real-time alerting system"""
    
    def __init__(self, config, event_bus):
        self.config = config
        self.event_bus = event_bus
        self.alert_rules = self._load_alert_rules(config['alert_rules'])
        self.active_alerts = {}
        self.alert_history = deque(maxlen=10000)
        
        # Alert channels
        self.alert_channels = self._initialize_alert_channels(config['channels'])
    
    async def evaluate_alerts(self, metric_name: str, value: float, tags: Dict[str, str]):
        """Evaluate metric against alert rules"""
        
        for rule_name, rule in self.alert_rules.items():
            if self._metric_matches_rule(metric_name, tags, rule):
                if self._should_trigger_alert(rule, value):
                    await self._trigger_alert(rule_name, rule, value, tags)
                elif self._should_resolve_alert(rule_name, value):
                    await self._resolve_alert(rule_name)
    
    async def _trigger_alert(self, rule_name: str, rule: Dict, value: float, tags: Dict):
        """Trigger an alert"""
        
        # Prevent alert spam
        if rule_name in self.active_alerts:
            last_alert_time = self.active_alerts[rule_name]['timestamp']
            if time.time() - last_alert_time < rule.get('cooldown', 300):  # 5min default
                return
        
        alert = {
            'id': str(uuid.uuid4()),
            'rule_name': rule_name,
            'severity': rule['severity'],
            'message': rule['message'].format(value=value, **tags),
            'value': value,
            'tags': tags,
            'timestamp': time.time(),
            'status': 'ACTIVE'
        }
        
        # Store active alert
        self.active_alerts[rule_name] = alert
        self.alert_history.append(alert)
        
        # Send notifications
        for channel_name in rule.get('channels', ['default']):
            channel = self.alert_channels.get(channel_name)
            if channel:
                await channel.send_alert(alert)
        
        # Publish alert event
        await self.event_bus.publish(Event(
            type=EventType.ALERT_TRIGGERED,
            payload=alert,
            source='alert_manager'
        ))
```

## Configuration

### Production Configuration

```yaml
monitoring:
  # Health monitoring
  health_monitor:
    check_interval: 30          # Global health check interval (seconds)
    component_timeout: 5        # Component health check timeout
    enable_auto_recovery: true  # Enable automatic recovery attempts
    recovery_attempts: 3        # Maximum recovery attempts
    
    # Component-specific settings
    components:
      strategic_agent:
        critical: true
        check_interval: 15
        timeout: 3
        recovery_strategy: "restart"
      
      tactical_agent:
        critical: true
        check_interval: 5
        timeout: 1
        recovery_strategy: "restart"
      
      execution_engine:
        critical: true
        check_interval: 1
        timeout: 0.5
        recovery_strategy: "failover"
      
      market_data_handler:
        critical: true
        check_interval: 10
        timeout: 2
        recovery_strategy: "reconnect"
  
  # Metrics collection
  metrics:
    collection_interval: 10     # Metrics collection interval (seconds)
    retention_hours: 168        # 7 days retention
    batch_size: 1000
    
    exporters:
      prometheus:
        enabled: true
        port: 9090
        path: "/metrics"
      
      influxdb:
        enabled: true
        host: "localhost"
        port: 8086
        database: "grandmodel"
        retention_policy: "7d"
      
      json_file:
        enabled: true
        file_path: "metrics/current_metrics.json"
        rotation_size: "100MB"
  
  # Logging
  logging:
    level: "INFO"
    format: "structured"
    
    handlers:
      - type: "rotating_file"
        filename: "logs/grandmodel.log"
        max_bytes: 100_000_000   # 100MB
        backup_count: 10
      
      - type: "console"
        level: "WARNING"
      
      - type: "syslog"
        address: ["localhost", 514]
        facility: "daemon"
    
    performance:
      async_logging: true
      buffer_size: 10000
      flush_interval: 1.0
  
  # Alerting
  alerting:
    enabled: true
    
    channels:
      email:
        enabled: true
        smtp_server: "localhost"
        smtp_port: 587
        recipients: ["ops@trading.com"]
      
      slack:
        enabled: true
        webhook_url: "${SLACK_WEBHOOK_URL}"
        channel: "#trading-alerts"
      
      pagerduty:
        enabled: true
        integration_key: "${PAGERDUTY_KEY}"
    
    rules:
      high_latency:
        metric: "processing_latency_ms"
        condition: "> 100"
        severity: "WARNING"
        message: "High processing latency detected: {value:.1f}ms"
        channels: ["slack"]
        cooldown: 300
      
      component_failure:
        metric: "component_health"
        condition: "== FAILED"
        severity: "CRITICAL"
        message: "Component failure detected: {component}"
        channels: ["email", "pagerduty"]
        cooldown: 60
      
      memory_usage:
        metric: "memory_usage_percent"
        condition: "> 85"
        severity: "WARNING"
        message: "High memory usage: {value:.1f}%"
        channels: ["slack"]
        cooldown: 600
      
      error_rate:
        metric: "error_rate_per_minute"
        condition: "> 10"
        severity: "CRITICAL"
        message: "High error rate: {value:.1f} errors/minute"
        channels: ["email", "slack"]
        cooldown: 180

  # Tactical monitoring (high-frequency)
  tactical:
    enabled: true
    check_interval_ms: 100      # 100ms checks
    
    latency_thresholds:
      warning: 10               # 10ms
      critical: 50              # 50ms
    
    components:
      - tactical_matrix_assembler
      - tactical_fvg_detector
      - order_router
      - execution_engine
```

### Development Configuration

```yaml
monitoring:
  health_monitor:
    check_interval: 10          # More frequent checks for development
    enable_debug_logging: true
    
  logging:
    level: "DEBUG"
    
    handlers:
      - type: "console"
        level: "DEBUG"
      
      - type: "file"
        filename: "logs/debug.log"
    
    # Save all metrics for analysis
    save_raw_metrics: true
    metrics_output_dir: "debug/metrics"
  
  alerting:
    # Disable external notifications in development
    channels:
      console:
        enabled: true
```

## Usage Examples

### System Health Monitoring

```python
from src.monitoring.health_monitor import HealthMonitor

# Initialize and start health monitoring
health_monitor = HealthMonitor(config, event_bus)
health_monitor.start()

# Check individual component health
strategic_agent_health = health_monitor.get_component_health('strategic_agent')
print(f"Strategic Agent: {strategic_agent_health.status}")
print(f"Last Check: {strategic_agent_health.last_check}")
print(f"Response Time: {strategic_agent_health.response_time_ms}ms")

# Get overall system health
system_health = health_monitor.get_system_health()
print(f"System Status: {system_health.overall_status}")
print(f"Components: {system_health.healthy_count}/{system_health.total_count} healthy")

# Subscribe to health changes
@event_bus.subscribe(EventType.HEALTH_STATUS_CHANGED)
async def on_health_change(event):
    health_data = event.payload
    if health_data['new_status'] == 'FAILED':
        print(f"ALERT: {health_data['component']} has failed!")
```

### Performance Metrics Collection

```python
from src.monitoring.metrics_exporter import MetricsExporter

# Initialize metrics collection
metrics = MetricsExporter(config)

# Record business metrics
metrics.increment_counter('trades_executed', tags={'strategy': 'marl'})
metrics.set_gauge('portfolio_value_usd', 1_000_000.50)
metrics.observe_histogram('order_fill_time_ms', 25.5)

# Record system metrics
metrics.set_gauge('cpu_usage_percent', 45.2)
metrics.set_gauge('memory_usage_mb', 512.8)
metrics.observe_histogram('gc_duration_ms', 12.3)

# Export metrics (happens automatically on schedule)
await metrics.export_metrics()

# Get current metric values
current_metrics = metrics.get_current_metrics()
print(f"Trades executed today: {current_metrics['trades_executed']}")
print(f"Average fill time: {current_metrics['avg_order_fill_time_ms']:.1f}ms")
```

### Custom Health Checks

```python
class CustomComponentHealthCheck:
    """Custom health check for specialized components"""
    
    def __init__(self, component_name, config):
        self.component_name = component_name
        self.config = config
        
    async def check_health(self) -> Dict[str, Any]:
        """Perform custom health check"""
        
        health_result = {
            'status': 'HEALTHY',
            'checks': {},
            'metrics': {},
            'timestamp': time.time()
        }
        
        try:
            # Check database connectivity
            db_health = await self._check_database_connection()
            health_result['checks']['database'] = db_health
            
            # Check external API connectivity
            api_health = await self._check_external_apis()
            health_result['checks']['external_apis'] = api_health
            
            # Check memory usage
            memory_usage = self._check_memory_usage()
            health_result['metrics']['memory_usage_mb'] = memory_usage
            health_result['checks']['memory'] = {
                'status': 'HEALTHY' if memory_usage < 1000 else 'WARNING'
            }
            
            # Check processing queue sizes
            queue_sizes = await self._check_queue_sizes()
            health_result['metrics']['queue_sizes'] = queue_sizes
            health_result['checks']['queues'] = {
                'status': 'HEALTHY' if max(queue_sizes.values()) < 1000 else 'WARNING'
            }
            
            # Determine overall status
            failed_checks = [
                name for name, check in health_result['checks'].items()
                if check['status'] == 'FAILED'
            ]
            
            if failed_checks:
                health_result['status'] = 'FAILED'
            elif any(check['status'] == 'WARNING' for check in health_result['checks'].values()):
                health_result['status'] = 'WARNING'
                
        except Exception as e:
            health_result['status'] = 'FAILED'
            health_result['error'] = str(e)
        
        return health_result

# Register custom health check
health_monitor.register_custom_health_check('my_component', CustomComponentHealthCheck)
```

## Performance Optimization

### Efficient Metrics Collection

```python
class HighPerformanceMetricsCollector:
    """Optimized metrics collector for high-frequency environments"""
    
    def __init__(self, config):
        self.config = config
        
        # Use lock-free data structures
        self.counters = {}
        self.gauges = {}
        self.histograms = {}
        
        # Batch processing
        self.batch_size = config.get('batch_size', 1000)
        self.pending_metrics = []
        
        # Memory-mapped storage for large datasets
        self.storage = MemoryMappedMetricsStorage(config['storage_path'])
    
    def record_metric_fast(self, metric_type: str, name: str, value: float, tags: Dict[str, str] = None):
        """High-performance metric recording"""
        
        # Use thread-local storage to avoid locks
        thread_local = threading.local()
        if not hasattr(thread_local, 'metric_buffer'):
            thread_local.metric_buffer = []
        
        # Buffer metrics for batch processing
        metric_record = {
            'type': metric_type,
            'name': name,
            'value': value,
            'tags': tags or {},
            'timestamp': time.time()
        }
        
        thread_local.metric_buffer.append(metric_record)
        
        # Flush buffer when full
        if len(thread_local.metric_buffer) >= self.batch_size:
            self._flush_metric_buffer(thread_local.metric_buffer)
            thread_local.metric_buffer.clear()
    
    def _flush_metric_buffer(self, buffer: List[Dict]):
        """Flush metric buffer to storage"""
        # Batch write to storage
        self.storage.write_batch(buffer)
        
        # Update in-memory aggregates
        for metric in buffer:
            self._update_aggregates(metric)
```

### Asynchronous Health Monitoring

```python
class AsyncHealthMonitor:
    """Asynchronous health monitoring for minimal performance impact"""
    
    def __init__(self, config, event_bus):
        self.config = config
        self.event_bus = event_bus
        self.component_checks = {}
        
        # Use asyncio for concurrent health checks
        self.check_semaphore = asyncio.Semaphore(config.get('max_concurrent_checks', 10))
    
    async def start_monitoring(self):
        """Start asynchronous health monitoring"""
        
        # Schedule health checks for all components
        tasks = []
        
        for component_name, check_config in self.config['components'].items():
            task = asyncio.create_task(
                self._monitor_component(component_name, check_config)
            )
            tasks.append(task)
        
        # Wait for all monitoring tasks
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _monitor_component(self, component_name: str, config: Dict):
        """Monitor individual component asynchronously"""
        
        check_interval = config['check_interval']
        
        while self.monitoring_active:
            async with self.check_semaphore:
                try:
                    # Perform health check with timeout
                    health_result = await asyncio.wait_for(
                        self._perform_health_check(component_name),
                        timeout=config.get('timeout', 5.0)
                    )
                    
                    # Update component status
                    await self._update_component_status(component_name, health_result)
                    
                except asyncio.TimeoutError:
                    await self._handle_health_check_timeout(component_name)
                except Exception as e:
                    await self._handle_health_check_error(component_name, e)
            
            # Wait for next check
            await asyncio.sleep(check_interval)
```

## Testing

### Unit Tests

```python
# tests/unit/test_monitoring/test_health_monitor.py
import pytest
from unittest.mock import Mock, AsyncMock
from src.monitoring.health_monitor import HealthMonitor

class TestHealthMonitor:
    def setUp(self):
        self.config = {
            'check_interval': 1,
            'components': {
                'test_component': {
                    'critical': True,
                    'timeout': 1
                }
            }
        }
        self.mock_event_bus = Mock()
        self.health_monitor = HealthMonitor(self.config, self.mock_event_bus)
    
    @pytest.mark.asyncio
    async def test_component_health_check(self):
        """Test component health check functionality"""
        
        # Mock component that returns healthy status
        mock_component = Mock()
        mock_component.get_health_status = AsyncMock(return_value={
            'status': 'HEALTHY',
            'response_time_ms': 5.0
        })
        
        # Register mock component
        self.health_monitor.register_component('test_component', mock_component)
        
        # Perform health check
        health_result = await self.health_monitor.check_component_health('test_component')
        
        assert health_result['status'] == 'HEALTHY'
        assert health_result['response_time_ms'] == 5.0
        mock_component.get_health_status.assert_called_once()
    
    def test_system_health_aggregation(self):
        """Test system health aggregation logic"""
        
        # Set up component health states
        self.health_monitor.component_health = {
            'component1': {'status': 'HEALTHY'},
            'component2': {'status': 'WARNING'},
            'component3': {'status': 'HEALTHY'}
        }
        
        system_health = self.health_monitor.get_system_health()
        
        assert system_health.overall_status == 'WARNING'  # Degraded due to one warning
        assert system_health.healthy_count == 2
        assert system_health.total_count == 3
```

### Integration Tests

```python
@pytest.mark.integration
def test_monitoring_integration():
    """Test complete monitoring system integration"""
    
    # Initialize monitoring system
    monitoring_system = MonitoringSystem(production_config, event_bus)
    monitoring_system.start()
    
    # Generate test load
    test_components = create_test_components()
    
    # Register components
    for component in test_components:
        monitoring_system.register_component(component.name, component)
    
    # Wait for monitoring cycles
    time.sleep(10)
    
    # Verify monitoring data
    system_health = monitoring_system.get_system_health()
    assert system_health.total_count == len(test_components)
    
    metrics = monitoring_system.get_current_metrics()
    assert 'component_health_checks_total' in metrics
    assert metrics['component_health_checks_total'] > 0
```

## Troubleshooting

### Common Issues

**High Monitoring Overhead:**
- Reduce health check frequency for non-critical components
- Use asynchronous health checks
- Implement health check caching
- Optimize metrics collection batch sizes

**Missing Health Data:**
- Check component registration
- Verify health check implementations
- Review timeout configurations
- Check network connectivity for remote components

**Alert Fatigue:**
- Implement alert cooldown periods
- Use alert severity levels
- Group related alerts
- Tune alert thresholds

### Debug Commands

```bash
# Check monitoring system health
curl http://localhost:8000/monitoring/health

# Get current metrics
curl http://localhost:8000/monitoring/metrics

# View active alerts
curl http://localhost:8000/monitoring/alerts

# Get component health details
curl http://localhost:8000/monitoring/components/strategic_agent

# Performance diagnostics
python -c "
from src.monitoring.health_monitor import HealthMonitor
monitor = HealthMonitor(config, None)
print(monitor.get_performance_diagnostics())
"
```

## Related Documentation

- [Core Components](../core/README.md)
- [System Architecture](../../docs/architecture/system_overview.md)
- [Deployment Guide](../../docs/guides/deployment_guide.md)
- [Performance Optimization](../../docs/guides/performance_guide.md)