# 📊 MONITORING AND ALERTING DOCUMENTATION
**COMPREHENSIVE OBSERVABILITY FOR SOLID FOUNDATION**

---

## 📋 EXECUTIVE SUMMARY

This comprehensive documentation provides detailed procedures for monitoring and alerting across all components of the SOLID FOUNDATION system. It covers metrics collection, alerting rules, dashboard configuration, and observability best practices to ensure system reliability and performance.

**Document Status**: MONITORING CRITICAL  
**Last Updated**: July 15, 2025  
**Target Audience**: SRE, Operations, Development Teams  
**Classification**: OPERATIONAL EXCELLENCE  

---

## 🎯 MONITORING STRATEGY

### Monitoring Pillars
```yaml
monitoring_pillars:
  metrics:
    description: "Quantitative measurements of system behavior"
    tools: ["Prometheus", "Grafana", "Custom metrics"]
    retention: "90 days"
    
  logs:
    description: "Structured and unstructured event records"
    tools: ["ELK Stack", "Fluentd", "Centralized logging"]
    retention: "30 days"
    
  traces:
    description: "Request flow through distributed systems"
    tools: ["Jaeger", "Zipkin", "OpenTelemetry"]
    retention: "7 days"
    
  events:
    description: "Discrete occurrences in the system"
    tools: ["Event Bus", "Audit logging", "Real-time streaming"]
    retention: "365 days"
```

### Observability Goals
```yaml
observability_goals:
  availability:
    target: "99.9%"
    measurement: "Uptime monitoring"
    alerting: "Immediate notification on downtime"
    
  performance:
    target: "< 100ms response time"
    measurement: "Request latency percentiles"
    alerting: "P95 > 100ms for 5 minutes"
    
  reliability:
    target: "< 0.1% error rate"
    measurement: "Error rate monitoring"
    alerting: "Error rate > 0.1% for 2 minutes"
    
  capacity:
    target: "< 80% resource utilization"
    measurement: "CPU, memory, disk usage"
    alerting: "Resource usage > 80% for 10 minutes"
```

---

## 📈 METRICS COLLECTION

### 1. PROMETHEUS CONFIGURATION

#### Prometheus Setup
```yaml
# /home/QuantNova/GrandModel/configs/monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'grandmodel-production'
    environment: 'production'

rule_files:
  - "/etc/prometheus/rules/*.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  # Application metrics
  - job_name: 'grandmodel-strategic'
    static_configs:
      - targets: ['grandmodel-strategic:9090']
    metrics_path: '/metrics'
    scrape_interval: 15s
    scrape_timeout: 10s
    
  - job_name: 'grandmodel-tactical'
    static_configs:
      - targets: ['grandmodel-tactical:9090']
    metrics_path: '/metrics'
    scrape_interval: 5s   # More frequent for tactical
    scrape_timeout: 3s
    
  - job_name: 'grandmodel-risk'
    static_configs:
      - targets: ['grandmodel-risk:9090']
    metrics_path: '/metrics'
    scrape_interval: 10s
    scrape_timeout: 5s
    
  # Infrastructure metrics
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['localhost:9100']
    
  - job_name: 'postgresql'
    static_configs:
      - targets: ['postgres-exporter:9187']
    
  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']
    
  # Kubernetes metrics
  - job_name: 'kubernetes-apiservers'
    kubernetes_sd_configs:
      - role: endpoints
    scheme: https
    tls_config:
      ca_file: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
    bearer_token_file: /var/run/secrets/kubernetes.io/serviceaccount/token
    relabel_configs:
      - source_labels: [__meta_kubernetes_namespace, __meta_kubernetes_service_name, __meta_kubernetes_endpoint_port_name]
        action: keep
        regex: default;kubernetes;https
    
  - job_name: 'kubernetes-nodes'
    kubernetes_sd_configs:
      - role: node
    scheme: https
    tls_config:
      ca_file: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
    bearer_token_file: /var/run/secrets/kubernetes.io/serviceaccount/token
    relabel_configs:
      - action: labelmap
        regex: __meta_kubernetes_node_label_(.+)
      - target_label: __address__
        replacement: kubernetes.default.svc:443
      - source_labels: [__meta_kubernetes_node_name]
        regex: (.+)
        target_label: __metrics_path__
        replacement: /api/v1/nodes/${1}/proxy/metrics
    
  - job_name: 'kubernetes-pods'
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)
      - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
        action: replace
        regex: ([^:]+)(?::\d+)?;(\d+)
        replacement: $1:$2
        target_label: __address__
      - action: labelmap
        regex: __meta_kubernetes_pod_label_(.+)
      - source_labels: [__meta_kubernetes_namespace]
        action: replace
        target_label: kubernetes_namespace
      - source_labels: [__meta_kubernetes_pod_name]
        action: replace
        target_label: kubernetes_pod_name
```

#### Custom Metrics Implementation
```python
# /home/QuantNova/GrandModel/src/monitoring/custom_metrics.py
from prometheus_client import Counter, Histogram, Gauge, Summary
import time
import functools
from typing import Dict, Any, Callable
import logging

class MetricsCollector:
    def __init__(self):
        # Application metrics
        self.inference_requests = Counter(
            'inference_requests_total',
            'Total inference requests',
            ['component', 'model', 'status']
        )
        
        self.inference_duration = Histogram(
            'inference_duration_seconds',
            'Inference duration in seconds',
            ['component', 'model'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]
        )
        
        self.active_connections = Gauge(
            'active_connections',
            'Number of active connections',
            ['component', 'connection_type']
        )
        
        self.queue_size = Gauge(
            'queue_size',
            'Current queue size',
            ['component', 'queue_type']
        )
        
        self.error_rate = Gauge(
            'error_rate',
            'Current error rate',
            ['component', 'error_type']
        )
        
        # Business metrics
        self.trades_executed = Counter(
            'trades_executed_total',
            'Total trades executed',
            ['symbol', 'strategy', 'side']
        )
        
        self.portfolio_value = Gauge(
            'portfolio_value_usd',
            'Portfolio value in USD',
            ['strategy']
        )
        
        self.risk_score = Gauge(
            'risk_score',
            'Current risk score',
            ['portfolio', 'risk_type']
        )
        
        self.pnl = Gauge(
            'pnl_usd',
            'Profit and Loss in USD',
            ['strategy', 'symbol']
        )
        
        # System metrics
        self.cpu_usage = Gauge(
            'cpu_usage_percent',
            'CPU usage percentage',
            ['component']
        )
        
        self.memory_usage = Gauge(
            'memory_usage_bytes',
            'Memory usage in bytes',
            ['component']
        )
        
        self.disk_usage = Gauge(
            'disk_usage_bytes',
            'Disk usage in bytes',
            ['component', 'mount_point']
        )
    
    def record_inference_request(self, component: str, model: str, 
                               status: str, duration: float = None):
        """Record inference request metrics"""
        self.inference_requests.labels(
            component=component, 
            model=model, 
            status=status
        ).inc()
        
        if duration is not None:
            self.inference_duration.labels(
                component=component, 
                model=model
            ).observe(duration)
    
    def update_connection_count(self, component: str, 
                              connection_type: str, count: int):
        """Update active connection count"""
        self.active_connections.labels(
            component=component, 
            connection_type=connection_type
        ).set(count)
    
    def update_queue_size(self, component: str, queue_type: str, size: int):
        """Update queue size"""
        self.queue_size.labels(
            component=component, 
            queue_type=queue_type
        ).set(size)
    
    def record_trade(self, symbol: str, strategy: str, side: str, 
                    value: float = None):
        """Record trade execution"""
        self.trades_executed.labels(
            symbol=symbol, 
            strategy=strategy, 
            side=side
        ).inc()
        
        if value is not None:
            # Update portfolio value
            self.portfolio_value.labels(strategy=strategy).set(value)
    
    def update_risk_score(self, portfolio: str, risk_type: str, score: float):
        """Update risk score"""
        self.risk_score.labels(
            portfolio=portfolio, 
            risk_type=risk_type
        ).set(score)
    
    def update_pnl(self, strategy: str, symbol: str, pnl: float):
        """Update PnL"""
        self.pnl.labels(strategy=strategy, symbol=symbol).set(pnl)
    
    def monitor_function(self, component: str, model: str = None):
        """Decorator to monitor function execution"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                status = 'success'
                
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    status = 'error'
                    logging.error(f"Function {func.__name__} failed: {e}")
                    raise
                finally:
                    duration = time.time() - start_time
                    self.record_inference_request(
                        component=component,
                        model=model or func.__name__,
                        status=status,
                        duration=duration
                    )
            
            return wrapper
        return decorator
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get current metrics summary"""
        return {
            'inference_requests': self.inference_requests._value._value,
            'active_connections': {
                metric.labels: metric._value._value 
                for metric in self.active_connections._metrics.values()
            },
            'queue_sizes': {
                metric.labels: metric._value._value 
                for metric in self.queue_size._metrics.values()
            },
            'portfolio_values': {
                metric.labels: metric._value._value 
                for metric in self.portfolio_value._metrics.values()
            }
        }

# Global metrics collector instance
metrics_collector = MetricsCollector()
```

### 2. APPLICATION METRICS

#### Strategic Component Metrics
```python
# /home/QuantNova/GrandModel/src/monitoring/strategic_metrics.py
from .custom_metrics import metrics_collector
import psutil
import time
import threading
from typing import Dict, Any

class StrategicMetricsCollector:
    def __init__(self):
        self.component_name = 'strategic'
        self.monitoring_active = False
        self.metrics_thread = None
        
    def start_monitoring(self):
        """Start metrics collection for strategic component"""
        self.monitoring_active = True
        self.metrics_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.metrics_thread.start()
        
    def stop_monitoring(self):
        """Stop metrics collection"""
        self.monitoring_active = False
        if self.metrics_thread:
            self.metrics_thread.join()
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # System metrics
                self._collect_system_metrics()
                
                # Application metrics
                self._collect_application_metrics()
                
                # Business metrics
                self._collect_business_metrics()
                
                time.sleep(15)  # Collect every 15 seconds
                
            except Exception as e:
                logging.error(f"Metrics collection error: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _collect_system_metrics(self):
        """Collect system-level metrics"""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        metrics_collector.cpu_usage.labels(
            component=self.component_name
        ).set(cpu_percent)
        
        # Memory usage
        memory = psutil.virtual_memory()
        metrics_collector.memory_usage.labels(
            component=self.component_name
        ).set(memory.used)
        
        # Disk usage
        disk = psutil.disk_usage('/')
        metrics_collector.disk_usage.labels(
            component=self.component_name,
            mount_point='/'
        ).set(disk.used)
    
    def _collect_application_metrics(self):
        """Collect application-specific metrics"""
        # This would integrate with actual application components
        # For now, simulate some metrics
        
        # Model inference metrics
        metrics_collector.update_queue_size(
            component=self.component_name,
            queue_type='inference',
            size=self._get_inference_queue_size()
        )
        
        # Connection metrics
        metrics_collector.update_connection_count(
            component=self.component_name,
            connection_type='database',
            count=self._get_database_connections()
        )
        
        metrics_collector.update_connection_count(
            component=self.component_name,
            connection_type='redis',
            count=self._get_redis_connections()
        )
    
    def _collect_business_metrics(self):
        """Collect business-related metrics"""
        # Portfolio metrics
        portfolio_value = self._get_portfolio_value()
        if portfolio_value is not None:
            metrics_collector.portfolio_value.labels(
                strategy='strategic'
            ).set(portfolio_value)
        
        # Risk metrics
        risk_score = self._calculate_risk_score()
        if risk_score is not None:
            metrics_collector.update_risk_score(
                portfolio='strategic',
                risk_type='var',
                score=risk_score
            )
    
    def _get_inference_queue_size(self) -> int:
        """Get current inference queue size"""
        # This would integrate with actual queue monitoring
        return 0
    
    def _get_database_connections(self) -> int:
        """Get database connection count"""
        # This would integrate with connection pool monitoring
        return 0
    
    def _get_redis_connections(self) -> int:
        """Get Redis connection count"""
        # This would integrate with Redis monitoring
        return 0
    
    def _get_portfolio_value(self) -> float:
        """Get current portfolio value"""
        # This would integrate with portfolio tracking
        return 0.0
    
    def _calculate_risk_score(self) -> float:
        """Calculate current risk score"""
        # This would integrate with risk calculation
        return 0.0

# Global strategic metrics collector
strategic_metrics = StrategicMetricsCollector()
```

### 3. INFRASTRUCTURE METRICS

#### System Metrics Configuration
```yaml
# /home/QuantNova/GrandModel/configs/monitoring/system_metrics.yaml
system_metrics:
  collection_interval: 15s
  retention_period: 90d
  
  cpu_metrics:
    - name: cpu_usage_percent
      description: "CPU usage percentage"
      labels: ["cpu", "mode"]
      thresholds:
        warning: 80
        critical: 95
    
    - name: cpu_load_average
      description: "CPU load average"
      labels: ["period"]
      thresholds:
        warning: 2.0
        critical: 4.0
  
  memory_metrics:
    - name: memory_usage_percent
      description: "Memory usage percentage"
      labels: ["type"]
      thresholds:
        warning: 85
        critical: 95
    
    - name: memory_available_bytes
      description: "Available memory in bytes"
      labels: ["type"]
      thresholds:
        warning: 1073741824  # 1GB
        critical: 536870912  # 512MB
  
  disk_metrics:
    - name: disk_usage_percent
      description: "Disk usage percentage"
      labels: ["device", "mount_point"]
      thresholds:
        warning: 85
        critical: 95
    
    - name: disk_io_operations
      description: "Disk I/O operations per second"
      labels: ["device", "operation"]
      thresholds:
        warning: 1000
        critical: 5000
  
  network_metrics:
    - name: network_bytes_total
      description: "Network bytes transferred"
      labels: ["device", "direction"]
      
    - name: network_packets_total
      description: "Network packets transferred"
      labels: ["device", "direction"]
    
    - name: network_errors_total
      description: "Network errors"
      labels: ["device", "type"]
```

---

## 🚨 ALERTING CONFIGURATION

### 1. ALERT RULES

#### Prometheus Alert Rules
```yaml
# /home/QuantNova/GrandModel/configs/monitoring/alerts.yml
groups:
  - name: system_alerts
    rules:
      - alert: HighCPUUsage
        expr: cpu_usage_percent > 80
        for: 5m
        labels:
          severity: warning
          component: system
        annotations:
          summary: "High CPU usage detected"
          description: "CPU usage is {{ $value }}% for component {{ $labels.component }}"
          runbook_url: "https://docs.company.com/runbooks/high-cpu"
      
      - alert: CriticalCPUUsage
        expr: cpu_usage_percent > 95
        for: 2m
        labels:
          severity: critical
          component: system
        annotations:
          summary: "Critical CPU usage detected"
          description: "CPU usage is {{ $value }}% for component {{ $labels.component }}"
          runbook_url: "https://docs.company.com/runbooks/critical-cpu"
      
      - alert: HighMemoryUsage
        expr: memory_usage_percent > 85
        for: 5m
        labels:
          severity: warning
          component: system
        annotations:
          summary: "High memory usage detected"
          description: "Memory usage is {{ $value }}% for component {{ $labels.component }}"
          runbook_url: "https://docs.company.com/runbooks/high-memory"
      
      - alert: CriticalMemoryUsage
        expr: memory_usage_percent > 95
        for: 2m
        labels:
          severity: critical
          component: system
        annotations:
          summary: "Critical memory usage detected"
          description: "Memory usage is {{ $value }}% for component {{ $labels.component }}"
          runbook_url: "https://docs.company.com/runbooks/critical-memory"
      
      - alert: HighDiskUsage
        expr: disk_usage_percent > 85
        for: 10m
        labels:
          severity: warning
          component: system
        annotations:
          summary: "High disk usage detected"
          description: "Disk usage is {{ $value }}% for mount point {{ $labels.mount_point }}"
          runbook_url: "https://docs.company.com/runbooks/high-disk"
      
      - alert: CriticalDiskUsage
        expr: disk_usage_percent > 95
        for: 5m
        labels:
          severity: critical
          component: system
        annotations:
          summary: "Critical disk usage detected"
          description: "Disk usage is {{ $value }}% for mount point {{ $labels.mount_point }}"
          runbook_url: "https://docs.company.com/runbooks/critical-disk"

  - name: application_alerts
    rules:
      - alert: ApplicationDown
        expr: up{job=~"grandmodel-.*"} == 0
        for: 1m
        labels:
          severity: critical
          component: application
        annotations:
          summary: "Application is down"
          description: "{{ $labels.job }} has been down for more than 1 minute"
          runbook_url: "https://docs.company.com/runbooks/application-down"
      
      - alert: HighErrorRate
        expr: rate(inference_requests_total{status="error"}[5m]) / rate(inference_requests_total[5m]) > 0.01
        for: 2m
        labels:
          severity: warning
          component: application
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }} for {{ $labels.component }}"
          runbook_url: "https://docs.company.com/runbooks/high-error-rate"
      
      - alert: CriticalErrorRate
        expr: rate(inference_requests_total{status="error"}[5m]) / rate(inference_requests_total[5m]) > 0.05
        for: 1m
        labels:
          severity: critical
          component: application
        annotations:
          summary: "Critical error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }} for {{ $labels.component }}"
          runbook_url: "https://docs.company.com/runbooks/critical-error-rate"
      
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(inference_duration_seconds_bucket[5m])) > 0.1
        for: 5m
        labels:
          severity: warning
          component: application
        annotations:
          summary: "High latency detected"
          description: "95th percentile latency is {{ $value }}s for {{ $labels.component }}"
          runbook_url: "https://docs.company.com/runbooks/high-latency"
      
      - alert: CriticalLatency
        expr: histogram_quantile(0.95, rate(inference_duration_seconds_bucket[5m])) > 0.5
        for: 2m
        labels:
          severity: critical
          component: application
        annotations:
          summary: "Critical latency detected"
          description: "95th percentile latency is {{ $value }}s for {{ $labels.component }}"
          runbook_url: "https://docs.company.com/runbooks/critical-latency"

  - name: business_alerts
    rules:
      - alert: HighRiskScore
        expr: risk_score > 0.8
        for: 5m
        labels:
          severity: warning
          component: risk
        annotations:
          summary: "High risk score detected"
          description: "Risk score is {{ $value }} for {{ $labels.portfolio }}"
          runbook_url: "https://docs.company.com/runbooks/high-risk"
      
      - alert: CriticalRiskScore
        expr: risk_score > 0.95
        for: 2m
        labels:
          severity: critical
          component: risk
        annotations:
          summary: "Critical risk score detected"
          description: "Risk score is {{ $value }} for {{ $labels.portfolio }}"
          runbook_url: "https://docs.company.com/runbooks/critical-risk"
      
      - alert: PortfolioLoss
        expr: pnl_usd < -10000
        for: 1m
        labels:
          severity: warning
          component: trading
        annotations:
          summary: "Portfolio loss detected"
          description: "Portfolio loss is ${{ $value }} for {{ $labels.strategy }}"
          runbook_url: "https://docs.company.com/runbooks/portfolio-loss"
      
      - alert: CriticalPortfolioLoss
        expr: pnl_usd < -50000
        for: 30s
        labels:
          severity: critical
          component: trading
        annotations:
          summary: "Critical portfolio loss detected"
          description: "Portfolio loss is ${{ $value }} for {{ $labels.strategy }}"
          runbook_url: "https://docs.company.com/runbooks/critical-portfolio-loss"

  - name: database_alerts
    rules:
      - alert: DatabaseDown
        expr: up{job="postgresql"} == 0
        for: 1m
        labels:
          severity: critical
          component: database
        annotations:
          summary: "Database is down"
          description: "PostgreSQL database has been down for more than 1 minute"
          runbook_url: "https://docs.company.com/runbooks/database-down"
      
      - alert: HighDatabaseConnections
        expr: pg_stat_database_numbackends > 180
        for: 5m
        labels:
          severity: warning
          component: database
        annotations:
          summary: "High database connections"
          description: "Database has {{ $value }} active connections"
          runbook_url: "https://docs.company.com/runbooks/high-db-connections"
      
      - alert: SlowDatabaseQueries
        expr: pg_stat_statements_mean_time_seconds > 1
        for: 5m
        labels:
          severity: warning
          component: database
        annotations:
          summary: "Slow database queries detected"
          description: "Average query time is {{ $value }}s"
          runbook_url: "https://docs.company.com/runbooks/slow-queries"
      
      - alert: RedisDown
        expr: up{job="redis"} == 0
        for: 1m
        labels:
          severity: critical
          component: cache
        annotations:
          summary: "Redis is down"
          description: "Redis cache has been down for more than 1 minute"
          runbook_url: "https://docs.company.com/runbooks/redis-down"
      
      - alert: HighRedisMemory
        expr: redis_memory_used_bytes / redis_memory_max_bytes > 0.9
        for: 5m
        labels:
          severity: warning
          component: cache
        annotations:
          summary: "High Redis memory usage"
          description: "Redis memory usage is {{ $value | humanizePercentage }}"
          runbook_url: "https://docs.company.com/runbooks/high-redis-memory"
```

### 2. ALERTMANAGER CONFIGURATION

#### Alertmanager Setup
```yaml
# /home/QuantNova/GrandModel/configs/monitoring/alertmanager.yml
global:
  smtp_smarthost: 'smtp.company.com:587'
  smtp_from: 'alerts@quantnova.com'
  smtp_auth_username: 'alerts@quantnova.com'
  smtp_auth_password: '${SMTP_PASSWORD}'

route:
  group_by: ['alertname', 'cluster', 'service']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'default-receiver'
  routes:
    - match:
        severity: critical
      receiver: 'critical-receiver'
      group_wait: 10s
      repeat_interval: 5m
    
    - match:
        severity: warning
      receiver: 'warning-receiver'
      group_wait: 30s
      repeat_interval: 30m
    
    - match:
        component: trading
      receiver: 'trading-receiver'
      group_wait: 5s
      repeat_interval: 2m
    
    - match:
        component: risk
      receiver: 'risk-receiver'
      group_wait: 5s
      repeat_interval: 1m

receivers:
  - name: 'default-receiver'
    email_configs:
      - to: 'ops-team@quantnova.com'
        subject: 'GrandModel Alert: {{ .GroupLabels.alertname }}'
        body: |
          {{ range .Alerts }}
          Alert: {{ .Annotations.summary }}
          Description: {{ .Annotations.description }}
          Labels: {{ range .Labels.SortedPairs }}{{ .Name }}: {{ .Value }}{{ end }}
          {{ end }}
  
  - name: 'critical-receiver'
    email_configs:
      - to: 'critical-alerts@quantnova.com'
        subject: 'CRITICAL: GrandModel Alert - {{ .GroupLabels.alertname }}'
        body: |
          CRITICAL ALERT
          
          {{ range .Alerts }}
          Alert: {{ .Annotations.summary }}
          Description: {{ .Annotations.description }}
          Severity: {{ .Labels.severity }}
          Component: {{ .Labels.component }}
          Runbook: {{ .Annotations.runbook_url }}
          {{ end }}
    
    slack_configs:
      - api_url: '${SLACK_WEBHOOK_URL}'
        channel: '#critical-alerts'
        title: 'CRITICAL: {{ .GroupLabels.alertname }}'
        text: |
          {{ range .Alerts }}
          {{ .Annotations.summary }}
          {{ .Annotations.description }}
          {{ end }}
        color: 'danger'
    
    pagerduty_configs:
      - service_key: '${PAGERDUTY_SERVICE_KEY}'
        description: 'CRITICAL: {{ .GroupLabels.alertname }}'
        details:
          summary: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
          component: '{{ .GroupLabels.component }}'
          severity: '{{ .GroupLabels.severity }}'
  
  - name: 'warning-receiver'
    email_configs:
      - to: 'warnings@quantnova.com'
        subject: 'WARNING: GrandModel Alert - {{ .GroupLabels.alertname }}'
        body: |
          {{ range .Alerts }}
          Alert: {{ .Annotations.summary }}
          Description: {{ .Annotations.description }}
          {{ end }}
    
    slack_configs:
      - api_url: '${SLACK_WEBHOOK_URL}'
        channel: '#alerts'
        title: 'WARNING: {{ .GroupLabels.alertname }}'
        text: |
          {{ range .Alerts }}
          {{ .Annotations.summary }}
          {{ end }}
        color: 'warning'
  
  - name: 'trading-receiver'
    email_configs:
      - to: 'trading-alerts@quantnova.com'
        subject: 'TRADING ALERT: {{ .GroupLabels.alertname }}'
        body: |
          {{ range .Alerts }}
          Alert: {{ .Annotations.summary }}
          Description: {{ .Annotations.description }}
          {{ end }}
    
    slack_configs:
      - api_url: '${SLACK_WEBHOOK_URL}'
        channel: '#trading-alerts'
        title: 'TRADING: {{ .GroupLabels.alertname }}'
        text: |
          {{ range .Alerts }}
          {{ .Annotations.summary }}
          {{ end }}
        color: 'danger'
  
  - name: 'risk-receiver'
    email_configs:
      - to: 'risk-alerts@quantnova.com'
        subject: 'RISK ALERT: {{ .GroupLabels.alertname }}'
        body: |
          {{ range .Alerts }}
          Alert: {{ .Annotations.summary }}
          Description: {{ .Annotations.description }}
          {{ end }}
    
    slack_configs:
      - api_url: '${SLACK_WEBHOOK_URL}'
        channel: '#risk-alerts'
        title: 'RISK: {{ .GroupLabels.alertname }}'
        text: |
          {{ range .Alerts }}
          {{ .Annotations.summary }}
          {{ end }}
        color: 'danger'

inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname', 'cluster', 'service']
```

### 3. CUSTOM ALERTING SYSTEM

#### Python Alert Manager
```python
# /home/QuantNova/GrandModel/src/monitoring/alert_manager.py
import smtplib
import json
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging

class AlertManager:
    def __init__(self):
        self.alert_rules = []
        self.alert_history = []
        self.notification_channels = {
            'email': self.send_email_alert,
            'slack': self.send_slack_alert,
            'webhook': self.send_webhook_alert,
            'pagerduty': self.send_pagerduty_alert
        }
        
        self.smtp_config = {
            'host': 'smtp.company.com',
            'port': 587,
            'username': 'alerts@quantnova.com',
            'password': '${SMTP_PASSWORD}',
            'from_email': 'alerts@quantnova.com'
        }
        
        self.slack_webhook = '${SLACK_WEBHOOK_URL}'
        self.pagerduty_key = '${PAGERDUTY_SERVICE_KEY}'
        
        self.alert_cooldown = 300  # 5 minutes
        self.active_alerts = {}
    
    def add_alert_rule(self, rule: Dict) -> None:
        """Add new alert rule"""
        required_fields = ['name', 'condition', 'severity', 'channels']
        
        for field in required_fields:
            if field not in rule:
                raise ValueError(f"Missing required field: {field}")
        
        self.alert_rules.append(rule)
    
    def evaluate_alerts(self, metrics: Dict) -> List[Dict]:
        """Evaluate all alert rules against current metrics"""
        triggered_alerts = []
        
        for rule in self.alert_rules:
            try:
                if self.evaluate_condition(rule['condition'], metrics):
                    # Check cooldown
                    if self.is_in_cooldown(rule['name']):
                        continue
                    
                    alert = self.create_alert(rule, metrics)
                    triggered_alerts.append(alert)
                    
                    # Send notifications
                    self.send_alert_notifications(alert)
                    
                    # Update cooldown
                    self.active_alerts[rule['name']] = datetime.now()
                    
            except Exception as e:
                logging.error(f"Error evaluating alert rule {rule['name']}: {e}")
        
        return triggered_alerts
    
    def evaluate_condition(self, condition: str, metrics: Dict) -> bool:
        """Evaluate alert condition"""
        # Simple condition evaluation
        # In production, this would be more sophisticated
        try:
            # Replace metric names with actual values
            for metric_name, value in metrics.items():
                condition = condition.replace(metric_name, str(value))
            
            # Evaluate the condition
            return eval(condition)
            
        except Exception as e:
            logging.error(f"Error evaluating condition: {condition}, {e}")
            return False
    
    def is_in_cooldown(self, alert_name: str) -> bool:
        """Check if alert is in cooldown period"""
        if alert_name not in self.active_alerts:
            return False
        
        last_triggered = self.active_alerts[alert_name]
        cooldown_end = last_triggered + timedelta(seconds=self.alert_cooldown)
        
        return datetime.now() < cooldown_end
    
    def create_alert(self, rule: Dict, metrics: Dict) -> Dict:
        """Create alert object"""
        alert = {
            'name': rule['name'],
            'severity': rule['severity'],
            'description': rule.get('description', ''),
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'runbook_url': rule.get('runbook_url', ''),
            'channels': rule['channels']
        }
        
        # Add to history
        self.alert_history.append(alert)
        
        return alert
    
    def send_alert_notifications(self, alert: Dict) -> None:
        """Send alert notifications to configured channels"""
        for channel in alert['channels']:
            try:
                if channel in self.notification_channels:
                    self.notification_channels[channel](alert)
                else:
                    logging.warning(f"Unknown notification channel: {channel}")
                    
            except Exception as e:
                logging.error(f"Failed to send alert to {channel}: {e}")
    
    def send_email_alert(self, alert: Dict) -> None:
        """Send email alert"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.smtp_config['from_email']
            msg['To'] = self.get_email_recipients(alert)
            msg['Subject'] = f"GrandModel Alert: {alert['name']}"
            
            body = self.format_email_body(alert)
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.smtp_config['host'], self.smtp_config['port'])
            server.starttls()
            server.login(self.smtp_config['username'], self.smtp_config['password'])
            server.send_message(msg)
            server.quit()
            
            logging.info(f"Email alert sent for {alert['name']}")
            
        except Exception as e:
            logging.error(f"Failed to send email alert: {e}")
    
    def send_slack_alert(self, alert: Dict) -> None:
        """Send Slack alert"""
        try:
            payload = {
                'text': f"GrandModel Alert: {alert['name']}",
                'attachments': [
                    {
                        'color': self.get_slack_color(alert['severity']),
                        'fields': [
                            {
                                'title': 'Description',
                                'value': alert['description'],
                                'short': False
                            },
                            {
                                'title': 'Severity',
                                'value': alert['severity'],
                                'short': True
                            },
                            {
                                'title': 'Timestamp',
                                'value': alert['timestamp'],
                                'short': True
                            }
                        ]
                    }
                ]
            }
            
            response = requests.post(self.slack_webhook, json=payload)
            response.raise_for_status()
            
            logging.info(f"Slack alert sent for {alert['name']}")
            
        except Exception as e:
            logging.error(f"Failed to send Slack alert: {e}")
    
    def send_webhook_alert(self, alert: Dict) -> None:
        """Send webhook alert"""
        try:
            webhook_url = alert.get('webhook_url')
            if not webhook_url:
                logging.warning("No webhook URL configured for alert")
                return
            
            payload = {
                'alert': alert,
                'timestamp': datetime.now().isoformat()
            }
            
            response = requests.post(webhook_url, json=payload)
            response.raise_for_status()
            
            logging.info(f"Webhook alert sent for {alert['name']}")
            
        except Exception as e:
            logging.error(f"Failed to send webhook alert: {e}")
    
    def send_pagerduty_alert(self, alert: Dict) -> None:
        """Send PagerDuty alert"""
        try:
            payload = {
                'routing_key': self.pagerduty_key,
                'event_action': 'trigger',
                'dedup_key': f"grandmodel_{alert['name']}",
                'payload': {
                    'summary': f"GrandModel Alert: {alert['name']}",
                    'source': 'grandmodel',
                    'severity': alert['severity'],
                    'custom_details': {
                        'description': alert['description'],
                        'metrics': alert['metrics'],
                        'runbook_url': alert.get('runbook_url', '')
                    }
                }
            }
            
            response = requests.post(
                'https://events.pagerduty.com/v2/enqueue',
                json=payload
            )
            response.raise_for_status()
            
            logging.info(f"PagerDuty alert sent for {alert['name']}")
            
        except Exception as e:
            logging.error(f"Failed to send PagerDuty alert: {e}")
    
    def get_email_recipients(self, alert: Dict) -> str:
        """Get email recipients based on alert severity"""
        severity_recipients = {
            'critical': 'critical-alerts@quantnova.com',
            'warning': 'warnings@quantnova.com',
            'info': 'info@quantnova.com'
        }
        
        return severity_recipients.get(alert['severity'], 'ops-team@quantnova.com')
    
    def get_slack_color(self, severity: str) -> str:
        """Get Slack color based on severity"""
        colors = {
            'critical': 'danger',
            'warning': 'warning',
            'info': 'good'
        }
        
        return colors.get(severity, 'good')
    
    def format_email_body(self, alert: Dict) -> str:
        """Format email body for alert"""
        body = f"""
GrandModel Alert: {alert['name']}

Severity: {alert['severity']}
Timestamp: {alert['timestamp']}
Description: {alert['description']}

Metrics:
{json.dumps(alert['metrics'], indent=2)}

Runbook: {alert.get('runbook_url', 'N/A')}

This is an automated alert from the GrandModel monitoring system.
"""
        return body
    
    def get_alert_summary(self, hours: int = 24) -> Dict:
        """Get alert summary for the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_alerts = [
            alert for alert in self.alert_history
            if datetime.fromisoformat(alert['timestamp']) > cutoff_time
        ]
        
        severity_counts = {}
        for alert in recent_alerts:
            severity = alert['severity']
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            'total_alerts': len(recent_alerts),
            'severity_breakdown': severity_counts,
            'time_range_hours': hours,
            'recent_alerts': recent_alerts[-10:]  # Last 10 alerts
        }

# Global alert manager instance
alert_manager = AlertManager()
```

---

## 📊 DASHBOARD CONFIGURATION

### 1. GRAFANA DASHBOARDS

#### Main Operations Dashboard
```json
{
  "dashboard": {
    "title": "GrandModel Operations Dashboard",
    "tags": ["grandmodel", "operations", "monitoring"],
    "timezone": "UTC",
    "refresh": "30s",
    "time": {
      "from": "now-6h",
      "to": "now"
    },
    "panels": [
      {
        "title": "System Overview",
        "type": "stat",
        "targets": [
          {
            "expr": "cpu_usage_percent",
            "legendFormat": "CPU Usage",
            "refId": "A"
          },
          {
            "expr": "memory_usage_percent",
            "legendFormat": "Memory Usage",
            "refId": "B"
          },
          {
            "expr": "disk_usage_percent",
            "legendFormat": "Disk Usage",
            "refId": "C"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "percent",
            "min": 0,
            "max": 100,
            "thresholds": {
              "steps": [
                {"color": "green", "value": 0},
                {"color": "yellow", "value": 70},
                {"color": "red", "value": 90}
              ]
            }
          }
        }
      },
      {
        "title": "Application Health",
        "type": "stat",
        "targets": [
          {
            "expr": "up{job=~\"grandmodel-.*\"}",
            "legendFormat": "{{ job }}",
            "refId": "A"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "mappings": [
              {"options": {"0": {"text": "Down", "color": "red"}}},
              {"options": {"1": {"text": "Up", "color": "green"}}}
            ]
          }
        }
      },
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(inference_requests_total[5m])",
            "legendFormat": "{{ component }} - {{ model }}",
            "refId": "A"
          }
        ],
        "yAxes": [
          {
            "label": "Requests/sec",
            "min": 0
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(inference_requests_total{status=\"error\"}[5m]) / rate(inference_requests_total[5m])",
            "legendFormat": "{{ component }} Error Rate",
            "refId": "A"
          }
        ],
        "yAxes": [
          {
            "label": "Error Rate",
            "min": 0,
            "max": 1
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(inference_duration_seconds_bucket[5m]))",
            "legendFormat": "{{ component }} P95",
            "refId": "A"
          },
          {
            "expr": "histogram_quantile(0.99, rate(inference_duration_seconds_bucket[5m]))",
            "legendFormat": "{{ component }} P99",
            "refId": "B"
          }
        ],
        "yAxes": [
          {
            "label": "Response Time (seconds)",
            "min": 0
          }
        ]
      },
      {
        "title": "Portfolio Metrics",
        "type": "graph",
        "targets": [
          {
            "expr": "portfolio_value_usd",
            "legendFormat": "{{ strategy }} Portfolio Value",
            "refId": "A"
          },
          {
            "expr": "pnl_usd",
            "legendFormat": "{{ strategy }} PnL",
            "refId": "B"
          }
        ],
        "yAxes": [
          {
            "label": "USD",
            "min": 0
          }
        ]
      },
      {
        "title": "Risk Metrics",
        "type": "graph",
        "targets": [
          {
            "expr": "risk_score",
            "legendFormat": "{{ portfolio }} Risk Score",
            "refId": "A"
          }
        ],
        "yAxes": [
          {
            "label": "Risk Score",
            "min": 0,
            "max": 1
          }
        ]
      }
    ]
  }
}
```

### 2. BUSINESS METRICS DASHBOARD

#### Trading Dashboard Configuration
```json
{
  "dashboard": {
    "title": "GrandModel Trading Dashboard",
    "tags": ["grandmodel", "trading", "business"],
    "timezone": "UTC",
    "refresh": "10s",
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "panels": [
      {
        "title": "Trading Volume",
        "type": "stat",
        "targets": [
          {
            "expr": "sum(rate(trades_executed_total[5m])) by (strategy)",
            "legendFormat": "{{ strategy }}",
            "refId": "A"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "short",
            "displayName": "Trades/min"
          }
        }
      },
      {
        "title": "Strategy Performance",
        "type": "table",
        "targets": [
          {
            "expr": "portfolio_value_usd",
            "format": "table",
            "refId": "A"
          },
          {
            "expr": "pnl_usd",
            "format": "table",
            "refId": "B"
          },
          {
            "expr": "risk_score",
            "format": "table",
            "refId": "C"
          }
        ],
        "transformations": [
          {
            "id": "merge",
            "options": {
              "reducers": []
            }
          }
        ]
      },
      {
        "title": "Risk Distribution",
        "type": "piechart",
        "targets": [
          {
            "expr": "risk_score",
            "legendFormat": "{{ portfolio }}",
            "refId": "A"
          }
        ]
      },
      {
        "title": "Market Impact",
        "type": "graph",
        "targets": [
          {
            "expr": "avg(inference_duration_seconds) by (component)",
            "legendFormat": "{{ component }} Latency",
            "refId": "A"
          }
        ]
      }
    ]
  }
}
```

---

## 📋 MONITORING CHECKLIST

### Daily Monitoring Tasks
```bash
#!/bin/bash
# Daily monitoring tasks

echo "=== Daily Monitoring Tasks ==="

# 1. Check system health
echo "1. System Health Check..."
python /home/QuantNova/GrandModel/src/monitoring/health_monitor.py --daily-check

# 2. Review alerts
echo "2. Alert Review..."
python /home/QuantNova/GrandModel/src/monitoring/alert_manager.py --alert-summary --hours=24

# 3. Performance review
echo "3. Performance Review..."
python /home/QuantNova/GrandModel/src/monitoring/performance_monitor.py --daily-report

# 4. Dashboard validation
echo "4. Dashboard Validation..."
curl -f http://localhost:3000/api/health || echo "Grafana dashboard unavailable"

# 5. Generate daily report
echo "5. Daily Report Generation..."
python /home/QuantNova/GrandModel/src/monitoring/daily_report_generator.py

echo "Daily monitoring tasks completed"
```

### Weekly Monitoring Tasks
```bash
#!/bin/bash
# Weekly monitoring tasks

echo "=== Weekly Monitoring Tasks ==="

# 1. Metrics retention cleanup
echo "1. Metrics Cleanup..."
python /home/QuantNova/GrandModel/src/monitoring/metrics_cleanup.py --retention-days=90

# 2. Alert rule validation
echo "2. Alert Rule Validation..."
python /home/QuantNova/GrandModel/src/monitoring/alert_validator.py --validate-all

# 3. Dashboard optimization
echo "3. Dashboard Optimization..."
python /home/QuantNova/GrandModel/src/monitoring/dashboard_optimizer.py --optimize-queries

# 4. Performance trending
echo "4. Performance Trending..."
python /home/QuantNova/GrandModel/src/monitoring/performance_trending.py --weekly-analysis

# 5. Generate weekly report
echo "5. Weekly Report Generation..."
python /home/QuantNova/GrandModel/src/monitoring/weekly_report_generator.py

echo "Weekly monitoring tasks completed"
```

---

## 🔧 MONITORING AUTOMATION

### Automated Monitoring Setup
```bash
#!/bin/bash
# Automated monitoring setup

echo "=== Automated Monitoring Setup ==="

# 1. Deploy Prometheus
echo "1. Deploying Prometheus..."
kubectl apply -f /home/QuantNova/GrandModel/k8s/monitoring/prometheus.yaml

# 2. Deploy Grafana
echo "2. Deploying Grafana..."
kubectl apply -f /home/QuantNova/GrandModel/k8s/monitoring/grafana.yaml

# 3. Deploy Alertmanager
echo "3. Deploying Alertmanager..."
kubectl apply -f /home/QuantNova/GrandModel/k8s/monitoring/alertmanager.yaml

# 4. Configure dashboards
echo "4. Configuring dashboards..."
python /home/QuantNova/GrandModel/src/monitoring/dashboard_provisioner.py --setup-all

# 5. Start monitoring services
echo "5. Starting monitoring services..."
python /home/QuantNova/GrandModel/src/monitoring/monitoring_manager.py --start-all

echo "Automated monitoring setup completed"
```

### Monitoring Health Check
```bash
#!/bin/bash
# Monitoring health check

echo "=== Monitoring Health Check ==="

# 1. Check Prometheus
echo "1. Prometheus Health:"
curl -f http://localhost:9090/-/healthy || echo "Prometheus unhealthy"

# 2. Check Grafana
echo "2. Grafana Health:"
curl -f http://localhost:3000/api/health || echo "Grafana unhealthy"

# 3. Check Alertmanager
echo "3. Alertmanager Health:"
curl -f http://localhost:9093/-/healthy || echo "Alertmanager unhealthy"

# 4. Check custom metrics
echo "4. Custom Metrics:"
python /home/QuantNova/GrandModel/src/monitoring/metrics_validator.py --validate-all

# 5. Alert system test
echo "5. Alert System Test:"
python /home/QuantNova/GrandModel/src/monitoring/alert_manager.py --test-alerts

echo "Monitoring health check completed"
```

---

**Document Version**: 1.0  
**Last Updated**: July 15, 2025  
**Next Review**: July 22, 2025  
**Owner**: SRE Team  
**Classification**: MONITORING CRITICAL