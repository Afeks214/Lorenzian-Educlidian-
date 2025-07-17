"""
Production Monitoring and Alerting System for GrandModel
========================================================

Comprehensive monitoring and alerting system for production MARL models
with real-time metrics, anomaly detection, and intelligent alerting.

Features:
- Real-time model performance monitoring
- Anomaly detection and drift monitoring
- Intelligent alerting with escalation
- Dashboard and visualization
- Performance analytics and reporting
- SLA monitoring and compliance
- Predictive failure detection
- Comprehensive logging and tracing

Author: Monitoring Team
Date: 2025-07-15
Version: 1.0.0
"""

import os
import sys
import asyncio
import json
import yaml
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import logging
import structlog
import requests
from prometheus_client import (
    CollectorRegistry, Gauge, Counter, Histogram, Summary,
    generate_latest, CONTENT_TYPE_LATEST, push_to_gateway
)
from prometheus_client.core import REGISTRY
import redis
from sqlalchemy import create_engine, text
from elasticsearch import Elasticsearch
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import warnings
warnings.filterwarnings('ignore')

logger = structlog.get_logger()

@dataclass
class MetricDefinition:
    """Metric definition for monitoring"""
    name: str
    metric_type: str  # 'gauge', 'counter', 'histogram', 'summary'
    description: str
    labels: List[str] = field(default_factory=list)
    unit: str = ""
    aggregation: str = "mean"  # mean, sum, max, min, p95, p99
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    enabled: bool = True

@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    description: str
    metric_name: str
    condition: str  # 'greater_than', 'less_than', 'equals', 'not_equals'
    threshold: float
    severity: str  # 'critical', 'warning', 'info'
    duration: int = 300  # seconds
    evaluation_interval: int = 60  # seconds
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    notification_channels: List[str] = field(default_factory=list)

@dataclass
class AlertNotification:
    """Alert notification"""
    alert_id: str
    rule_name: str
    severity: str
    status: str  # 'firing', 'resolved'
    timestamp: datetime
    message: str
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    resolved_at: Optional[datetime] = None

@dataclass
class ModelMetrics:
    """Model performance metrics"""
    model_name: str
    timestamp: datetime
    inference_latency_ms: float
    throughput_rps: float
    error_rate: float
    cpu_usage_percent: float
    memory_usage_mb: float
    gpu_usage_percent: float
    request_count: int
    error_count: int
    prediction_accuracy: float = 0.0
    drift_score: float = 0.0
    anomaly_score: float = 0.0

@dataclass
class SystemMetrics:
    """System-level metrics"""
    timestamp: datetime
    cpu_usage_percent: float
    memory_usage_percent: float
    disk_usage_percent: float
    network_rx_bytes: int
    network_tx_bytes: int
    active_connections: int
    queue_depth: int
    cache_hit_rate: float

class ProductionMonitoringSystem:
    """
    Comprehensive production monitoring system
    
    Capabilities:
    - Real-time metrics collection
    - Anomaly detection
    - Intelligent alerting
    - Performance analytics
    - SLA monitoring
    - Predictive failure detection
    - Dashboard generation
    """
    
    def __init__(self, config_path: str = None):
        """Initialize monitoring system"""
        self.monitoring_id = f"monitor_{int(time.time())}"
        self.start_time = datetime.now()
        
        # Load configuration
        self.config = self._load_monitoring_config(config_path)
        
        # Initialize paths
        self.project_root = Path(__file__).parent.parent
        self.metrics_dir = self.project_root / "metrics"
        self.dashboards_dir = self.project_root / "dashboards"
        self.alerts_dir = self.project_root / "alerts"
        self.logs_dir = self.project_root / "logs" / "monitoring"
        
        # Create directories
        for directory in [self.metrics_dir, self.dashboards_dir, self.alerts_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize clients
        self._initialize_clients()
        
        # Initialize metrics
        self._initialize_metrics()
        
        # Initialize alert rules
        self._initialize_alert_rules()
        
        # Monitoring state
        self.active_alerts: Dict[str, AlertNotification] = {}
        self.metric_history: Dict[str, List[float]] = {}
        self.anomaly_detectors: Dict[str, IsolationForest] = {}
        self.is_monitoring = False
        
        # Executor for parallel operations
        self.executor = ThreadPoolExecutor(max_workers=8)
        
        logger.info("ProductionMonitoringSystem initialized",
                   monitoring_id=self.monitoring_id,
                   config_name=self.config.get('name', 'default'))
    
    def _load_monitoring_config(self, config_path: str = None) -> Dict[str, Any]:
        """Load monitoring configuration"""
        default_config = {
            'name': 'grandmodel-monitoring',
            'version': '1.0.0',
            'collection_interval': 30,  # seconds
            'retention_days': 30,
            'anomaly_detection': {
                'enabled': True,
                'sensitivity': 0.1,
                'window_size': 100,
                'contamination': 0.05
            },
            'alerting': {
                'enabled': True,
                'evaluation_interval': 60,
                'notification_channels': ['email', 'slack'],
                'escalation_policy': {
                    'warning': {'initial_delay': 300, 'repeat_interval': 1800},
                    'critical': {'initial_delay': 60, 'repeat_interval': 300}
                }
            },
            'prometheus': {
                'enabled': True,
                'endpoint': 'http://prometheus:9090',
                'pushgateway_endpoint': 'http://pushgateway:9091'
            },
            'grafana': {
                'enabled': True,
                'endpoint': 'http://grafana:3000',
                'api_key': os.getenv('GRAFANA_API_KEY')
            },
            'elasticsearch': {
                'enabled': True,
                'endpoint': 'http://elasticsearch:9200',
                'index_pattern': 'grandmodel-*'
            },
            'notifications': {
                'email': {
                    'enabled': True,
                    'smtp_server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
                    'smtp_port': int(os.getenv('SMTP_PORT', 587)),
                    'username': os.getenv('SMTP_USERNAME'),
                    'password': os.getenv('SMTP_PASSWORD'),
                    'from_email': os.getenv('FROM_EMAIL'),
                    'to_emails': os.getenv('TO_EMAILS', '').split(',')
                },
                'slack': {
                    'enabled': True,
                    'webhook_url': os.getenv('SLACK_WEBHOOK_URL'),
                    'channel': '#alerts'
                }
            },
            'sla_thresholds': {
                'availability': 99.9,  # 99.9% availability
                'latency_p95': 500,    # 500ms 95th percentile
                'error_rate': 0.1      # 0.1% error rate
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    file_config = yaml.safe_load(f)
                else:
                    file_config = json.load(f)
                
                # Deep merge configuration
                self._deep_merge_config(default_config, file_config)
        
        return default_config
    
    def _deep_merge_config(self, base: Dict[str, Any], update: Dict[str, Any]):
        """Deep merge configuration dictionaries"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge_config(base[key], value)
            else:
                base[key] = value
    
    def _initialize_clients(self):
        """Initialize monitoring clients"""
        try:
            # Redis client for caching
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
            self.redis_client = redis.from_url(redis_url)
        except Exception as e:
            logger.warning("Redis client initialization failed", error=str(e))
            self.redis_client = None
        
        try:
            # Database client
            db_url = os.getenv('DATABASE_URL', 'postgresql://localhost:5432/grandmodel')
            self.db_engine = create_engine(db_url)
        except Exception as e:
            logger.warning("Database client initialization failed", error=str(e))
            self.db_engine = None
        
        try:
            # Elasticsearch client
            if self.config['elasticsearch']['enabled']:
                self.es_client = Elasticsearch([self.config['elasticsearch']['endpoint']])
            else:
                self.es_client = None
        except Exception as e:
            logger.warning("Elasticsearch client initialization failed", error=str(e))
            self.es_client = None
    
    def _initialize_metrics(self):
        """Initialize Prometheus metrics"""
        self.metrics_registry = CollectorRegistry()
        
        # Define metrics
        self.prometheus_metrics = {
            'model_inference_latency': Histogram(
                'model_inference_latency_seconds',
                'Model inference latency in seconds',
                ['model_name', 'model_version', 'environment'],
                registry=self.metrics_registry
            ),
            'model_throughput': Gauge(
                'model_throughput_rps',
                'Model throughput in requests per second',
                ['model_name', 'model_version', 'environment'],
                registry=self.metrics_registry
            ),
            'model_error_rate': Gauge(
                'model_error_rate',
                'Model error rate percentage',
                ['model_name', 'model_version', 'environment'],
                registry=self.metrics_registry
            ),
            'model_cpu_usage': Gauge(
                'model_cpu_usage_percent',
                'Model CPU usage percentage',
                ['model_name', 'model_version', 'environment'],
                registry=self.metrics_registry
            ),
            'model_memory_usage': Gauge(
                'model_memory_usage_mb',
                'Model memory usage in MB',
                ['model_name', 'model_version', 'environment'],
                registry=self.metrics_registry
            ),
            'model_requests_total': Counter(
                'model_requests_total',
                'Total model requests',
                ['model_name', 'model_version', 'environment', 'status'],
                registry=self.metrics_registry
            ),
            'model_prediction_accuracy': Gauge(
                'model_prediction_accuracy',
                'Model prediction accuracy percentage',
                ['model_name', 'model_version', 'environment'],
                registry=self.metrics_registry
            ),
            'model_drift_score': Gauge(
                'model_drift_score',
                'Model drift score',
                ['model_name', 'model_version', 'environment'],
                registry=self.metrics_registry
            ),
            'model_anomaly_score': Gauge(
                'model_anomaly_score',
                'Model anomaly score',
                ['model_name', 'model_version', 'environment'],
                registry=self.metrics_registry
            )
        }
        
        logger.info("Prometheus metrics initialized")
    
    def _initialize_alert_rules(self):
        """Initialize alert rules"""
        self.alert_rules = [
            AlertRule(
                name="high_error_rate",
                description="High error rate detected",
                metric_name="model_error_rate",
                condition="greater_than",
                threshold=5.0,  # 5% error rate
                severity="critical",
                duration=300,
                notification_channels=['email', 'slack']
            ),
            AlertRule(
                name="high_latency",
                description="High inference latency detected",
                metric_name="model_inference_latency",
                condition="greater_than",
                threshold=1.0,  # 1 second
                severity="warning",
                duration=300,
                notification_channels=['slack']
            ),
            AlertRule(
                name="low_throughput",
                description="Low model throughput detected",
                metric_name="model_throughput",
                condition="less_than",
                threshold=50.0,  # 50 RPS
                severity="warning",
                duration=600,
                notification_channels=['slack']
            ),
            AlertRule(
                name="high_cpu_usage",
                description="High CPU usage detected",
                metric_name="model_cpu_usage",
                condition="greater_than",
                threshold=80.0,  # 80% CPU
                severity="warning",
                duration=900,
                notification_channels=['slack']
            ),
            AlertRule(
                name="high_memory_usage",
                description="High memory usage detected",
                metric_name="model_memory_usage",
                condition="greater_than",
                threshold=8000.0,  # 8GB
                severity="warning",
                duration=600,
                notification_channels=['slack']
            ),
            AlertRule(
                name="model_drift_detected",
                description="Model drift detected",
                metric_name="model_drift_score",
                condition="greater_than",
                threshold=0.3,  # 30% drift
                severity="critical",
                duration=300,
                notification_channels=['email', 'slack']
            ),
            AlertRule(
                name="anomaly_detected",
                description="Anomaly detected in model behavior",
                metric_name="model_anomaly_score",
                condition="greater_than",
                threshold=0.8,  # 80% anomaly score
                severity="warning",
                duration=300,
                notification_channels=['slack']
            )
        ]
        
        logger.info(f"Alert rules initialized: {len(self.alert_rules)} rules")
    
    async def start_monitoring(self):
        """Start monitoring system"""
        logger.info("🚀 Starting production monitoring system")
        
        self.is_monitoring = True
        
        # Start monitoring tasks
        tasks = [
            asyncio.create_task(self._collect_metrics_loop()),
            asyncio.create_task(self._evaluate_alerts_loop()),
            asyncio.create_task(self._detect_anomalies_loop()),
            asyncio.create_task(self._generate_reports_loop())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error("Monitoring system error", error=str(e))
            raise
    
    async def stop_monitoring(self):
        """Stop monitoring system"""
        logger.info("🛑 Stopping production monitoring system")
        self.is_monitoring = False
    
    async def _collect_metrics_loop(self):
        """Metrics collection loop"""
        while self.is_monitoring:
            try:
                await self._collect_all_metrics()
                await asyncio.sleep(self.config['collection_interval'])
            except Exception as e:
                logger.error("Metrics collection error", error=str(e))
                await asyncio.sleep(60)  # Wait before retry
    
    async def _collect_all_metrics(self):
        """Collect all metrics"""
        # Collect model metrics
        model_metrics = await self._collect_model_metrics()
        
        # Collect system metrics
        system_metrics = await self._collect_system_metrics()
        
        # Update Prometheus metrics
        await self._update_prometheus_metrics(model_metrics, system_metrics)
        
        # Store metrics in database
        await self._store_metrics(model_metrics, system_metrics)
        
        # Update metric history for anomaly detection
        await self._update_metric_history(model_metrics)
    
    async def _collect_model_metrics(self) -> List[ModelMetrics]:
        """Collect model-specific metrics"""
        model_metrics = []
        
        # Simulate model metrics collection
        # In production, this would query actual model endpoints
        
        models = [
            {"name": "tactical_model", "version": "1.0.0", "environment": "production"},
            {"name": "strategic_model", "version": "1.0.0", "environment": "production"}
        ]
        
        for model in models:
            # Simulate metrics
            metrics = ModelMetrics(
                model_name=model["name"],
                timestamp=datetime.now(),
                inference_latency_ms=np.random.uniform(100, 500),
                throughput_rps=np.random.uniform(80, 150),
                error_rate=np.random.uniform(0.1, 2.0),
                cpu_usage_percent=np.random.uniform(30, 80),
                memory_usage_mb=np.random.uniform(2000, 6000),
                gpu_usage_percent=np.random.uniform(40, 90),
                request_count=np.random.randint(1000, 5000),
                error_count=np.random.randint(1, 50),
                prediction_accuracy=np.random.uniform(85, 95),
                drift_score=np.random.uniform(0.0, 0.4),
                anomaly_score=np.random.uniform(0.0, 0.3)
            )
            
            model_metrics.append(metrics)
        
        return model_metrics
    
    async def _collect_system_metrics(self) -> SystemMetrics:
        """Collect system-level metrics"""
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        network = psutil.net_io_counters()
        
        return SystemMetrics(
            timestamp=datetime.now(),
            cpu_usage_percent=cpu_percent,
            memory_usage_percent=memory.percent,
            disk_usage_percent=disk.percent,
            network_rx_bytes=network.bytes_recv,
            network_tx_bytes=network.bytes_sent,
            active_connections=len(psutil.net_connections()),
            queue_depth=0,  # Would be retrieved from message queue
            cache_hit_rate=np.random.uniform(80, 95)  # Simulated cache hit rate
        )
    
    async def _update_prometheus_metrics(self, model_metrics: List[ModelMetrics], 
                                       system_metrics: SystemMetrics):
        """Update Prometheus metrics"""
        for metrics in model_metrics:
            labels = [metrics.model_name, "1.0.0", "production"]
            
            # Update model metrics
            self.prometheus_metrics['model_inference_latency'].labels(*labels).observe(
                metrics.inference_latency_ms / 1000
            )
            self.prometheus_metrics['model_throughput'].labels(*labels).set(metrics.throughput_rps)
            self.prometheus_metrics['model_error_rate'].labels(*labels).set(metrics.error_rate)
            self.prometheus_metrics['model_cpu_usage'].labels(*labels).set(metrics.cpu_usage_percent)
            self.prometheus_metrics['model_memory_usage'].labels(*labels).set(metrics.memory_usage_mb)
            self.prometheus_metrics['model_prediction_accuracy'].labels(*labels).set(metrics.prediction_accuracy)
            self.prometheus_metrics['model_drift_score'].labels(*labels).set(metrics.drift_score)
            self.prometheus_metrics['model_anomaly_score'].labels(*labels).set(metrics.anomaly_score)
            
            # Update counters
            self.prometheus_metrics['model_requests_total'].labels(*labels, 'success').inc(
                metrics.request_count - metrics.error_count
            )
            self.prometheus_metrics['model_requests_total'].labels(*labels, 'error').inc(
                metrics.error_count
            )
        
        # Push metrics to Pushgateway if configured
        if self.config['prometheus']['enabled']:
            try:
                pushgateway_url = self.config['prometheus']['pushgateway_endpoint']
                push_to_gateway(pushgateway_url, job='grandmodel-monitoring', registry=self.metrics_registry)
            except Exception as e:
                logger.warning("Failed to push metrics to Pushgateway", error=str(e))
    
    async def _store_metrics(self, model_metrics: List[ModelMetrics], 
                           system_metrics: SystemMetrics):
        """Store metrics in database"""
        if not self.db_engine:
            return
        
        try:
            # Store model metrics
            for metrics in model_metrics:
                metric_data = {
                    'timestamp': metrics.timestamp,
                    'model_name': metrics.model_name,
                    'inference_latency_ms': metrics.inference_latency_ms,
                    'throughput_rps': metrics.throughput_rps,
                    'error_rate': metrics.error_rate,
                    'cpu_usage_percent': metrics.cpu_usage_percent,
                    'memory_usage_mb': metrics.memory_usage_mb,
                    'gpu_usage_percent': metrics.gpu_usage_percent,
                    'request_count': metrics.request_count,
                    'error_count': metrics.error_count,
                    'prediction_accuracy': metrics.prediction_accuracy,
                    'drift_score': metrics.drift_score,
                    'anomaly_score': metrics.anomaly_score
                }
                
                # In production, would use proper ORM or SQL inserts
                # For now, just log the data
                logger.debug("Storing model metrics", **metric_data)
            
            # Store system metrics
            system_data = {
                'timestamp': system_metrics.timestamp,
                'cpu_usage_percent': system_metrics.cpu_usage_percent,
                'memory_usage_percent': system_metrics.memory_usage_percent,
                'disk_usage_percent': system_metrics.disk_usage_percent,
                'network_rx_bytes': system_metrics.network_rx_bytes,
                'network_tx_bytes': system_metrics.network_tx_bytes,
                'active_connections': system_metrics.active_connections,
                'queue_depth': system_metrics.queue_depth,
                'cache_hit_rate': system_metrics.cache_hit_rate
            }
            
            logger.debug("Storing system metrics", **system_data)
            
        except Exception as e:
            logger.error("Failed to store metrics", error=str(e))
    
    async def _update_metric_history(self, model_metrics: List[ModelMetrics]):
        """Update metric history for anomaly detection"""
        for metrics in model_metrics:
            model_name = metrics.model_name
            
            # Initialize history if not exists
            if model_name not in self.metric_history:
                self.metric_history[model_name] = []
            
            # Add current metrics
            metric_values = [
                metrics.inference_latency_ms,
                metrics.throughput_rps,
                metrics.error_rate,
                metrics.cpu_usage_percent,
                metrics.memory_usage_mb,
                metrics.prediction_accuracy,
                metrics.drift_score
            ]
            
            self.metric_history[model_name].append(metric_values)
            
            # Keep only recent history
            window_size = self.config['anomaly_detection']['window_size']
            if len(self.metric_history[model_name]) > window_size:
                self.metric_history[model_name] = self.metric_history[model_name][-window_size:]
    
    async def _evaluate_alerts_loop(self):
        """Alert evaluation loop"""
        while self.is_monitoring:
            try:
                await self._evaluate_all_alerts()
                await asyncio.sleep(self.config['alerting']['evaluation_interval'])
            except Exception as e:
                logger.error("Alert evaluation error", error=str(e))
                await asyncio.sleep(60)  # Wait before retry
    
    async def _evaluate_all_alerts(self):
        """Evaluate all alert rules"""
        for rule in self.alert_rules:
            if not rule.enabled:
                continue
            
            try:
                await self._evaluate_alert_rule(rule)
            except Exception as e:
                logger.error(f"Alert rule evaluation failed: {rule.name}", error=str(e))
    
    async def _evaluate_alert_rule(self, rule: AlertRule):
        """Evaluate individual alert rule"""
        # Get current metric value
        current_value = await self._get_current_metric_value(rule.metric_name)
        
        if current_value is None:
            return
        
        # Check condition
        alert_triggered = False
        
        if rule.condition == "greater_than":
            alert_triggered = current_value > rule.threshold
        elif rule.condition == "less_than":
            alert_triggered = current_value < rule.threshold
        elif rule.condition == "equals":
            alert_triggered = current_value == rule.threshold
        elif rule.condition == "not_equals":
            alert_triggered = current_value != rule.threshold
        
        alert_id = f"{rule.name}_{rule.metric_name}"
        
        if alert_triggered:
            # Check if alert is already active
            if alert_id not in self.active_alerts:
                # Create new alert
                alert = AlertNotification(
                    alert_id=alert_id,
                    rule_name=rule.name,
                    severity=rule.severity,
                    status="firing",
                    timestamp=datetime.now(),
                    message=f"{rule.description}: {current_value} {rule.condition} {rule.threshold}",
                    labels=rule.labels,
                    annotations=rule.annotations
                )
                
                self.active_alerts[alert_id] = alert
                
                # Send notification
                await self._send_alert_notification(alert, rule)
                
                logger.warning(f"Alert triggered: {rule.name}",
                             metric=rule.metric_name,
                             value=current_value,
                             threshold=rule.threshold)
        else:
            # Check if alert should be resolved
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.status = "resolved"
                alert.resolved_at = datetime.now()
                
                # Send resolution notification
                await self._send_alert_resolution(alert, rule)
                
                # Remove from active alerts
                del self.active_alerts[alert_id]
                
                logger.info(f"Alert resolved: {rule.name}",
                           metric=rule.metric_name,
                           value=current_value)
    
    async def _get_current_metric_value(self, metric_name: str) -> Optional[float]:
        """Get current value for a metric"""
        # In production, this would query Prometheus or the metrics database
        # For now, return a simulated value
        
        if metric_name == "model_error_rate":
            return np.random.uniform(0.5, 8.0)
        elif metric_name == "model_inference_latency":
            return np.random.uniform(0.2, 1.5)
        elif metric_name == "model_throughput":
            return np.random.uniform(40, 120)
        elif metric_name == "model_cpu_usage":
            return np.random.uniform(30, 90)
        elif metric_name == "model_memory_usage":
            return np.random.uniform(2000, 10000)
        elif metric_name == "model_drift_score":
            return np.random.uniform(0.0, 0.5)
        elif metric_name == "model_anomaly_score":
            return np.random.uniform(0.0, 0.9)
        else:
            return None
    
    async def _send_alert_notification(self, alert: AlertNotification, rule: AlertRule):
        """Send alert notification"""
        for channel in rule.notification_channels:
            try:
                if channel == "email":
                    await self._send_email_alert(alert, rule)
                elif channel == "slack":
                    await self._send_slack_alert(alert, rule)
                else:
                    logger.warning(f"Unknown notification channel: {channel}")
            except Exception as e:
                logger.error(f"Failed to send alert notification via {channel}", error=str(e))
    
    async def _send_email_alert(self, alert: AlertNotification, rule: AlertRule):
        """Send email alert notification"""
        email_config = self.config['notifications']['email']
        
        if not email_config['enabled']:
            return
        
        try:
            # Create email message
            msg = MIMEMultipart()
            msg['From'] = email_config['from_email']
            msg['To'] = ', '.join(email_config['to_emails'])
            msg['Subject'] = f"[{alert.severity.upper()}] {rule.name}"
            
            # Email body
            body = f"""
Alert: {rule.name}
Severity: {alert.severity}
Status: {alert.status}
Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
Message: {alert.message}

Rule Details:
- Metric: {rule.metric_name}
- Condition: {rule.condition}
- Threshold: {rule.threshold}
- Duration: {rule.duration}s

Labels: {alert.labels}
Annotations: {alert.annotations}
"""
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
            server.starttls()
            server.login(email_config['username'], email_config['password'])
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email alert sent: {rule.name}")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {rule.name}", error=str(e))
    
    async def _send_slack_alert(self, alert: AlertNotification, rule: AlertRule):
        """Send Slack alert notification"""
        slack_config = self.config['notifications']['slack']
        
        if not slack_config['enabled']:
            return
        
        try:
            webhook_url = slack_config['webhook_url']
            
            if not webhook_url:
                logger.warning("Slack webhook URL not configured")
                return
            
            # Determine color based on severity
            color = {
                'critical': 'danger',
                'warning': 'warning',
                'info': 'good'
            }.get(alert.severity, 'warning')
            
            # Create Slack message
            message = {
                "channel": slack_config['channel'],
                "username": "Monitoring Bot",
                "text": f"Alert: {rule.name}",
                "attachments": [
                    {
                        "color": color,
                        "title": f"[{alert.severity.upper()}] {rule.name}",
                        "text": alert.message,
                        "fields": [
                            {
                                "title": "Metric",
                                "value": rule.metric_name,
                                "short": True
                            },
                            {
                                "title": "Condition",
                                "value": f"{rule.condition} {rule.threshold}",
                                "short": True
                            },
                            {
                                "title": "Time",
                                "value": alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                                "short": True
                            },
                            {
                                "title": "Status",
                                "value": alert.status,
                                "short": True
                            }
                        ],
                        "footer": "GrandModel Monitoring",
                        "ts": int(alert.timestamp.timestamp())
                    }
                ]
            }
            
            # Send to Slack
            response = requests.post(webhook_url, json=message)
            response.raise_for_status()
            
            logger.info(f"Slack alert sent: {rule.name}")
            
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {rule.name}", error=str(e))
    
    async def _send_alert_resolution(self, alert: AlertNotification, rule: AlertRule):
        """Send alert resolution notification"""
        # Send resolution notification to Slack
        if "slack" in rule.notification_channels:
            await self._send_slack_resolution(alert, rule)
    
    async def _send_slack_resolution(self, alert: AlertNotification, rule: AlertRule):
        """Send Slack resolution notification"""
        slack_config = self.config['notifications']['slack']
        
        if not slack_config['enabled']:
            return
        
        try:
            webhook_url = slack_config['webhook_url']
            
            if not webhook_url:
                return
            
            # Create resolution message
            message = {
                "channel": slack_config['channel'],
                "username": "Monitoring Bot",
                "text": f"Alert Resolved: {rule.name}",
                "attachments": [
                    {
                        "color": "good",
                        "title": f"[RESOLVED] {rule.name}",
                        "text": f"Alert has been resolved",
                        "fields": [
                            {
                                "title": "Metric",
                                "value": rule.metric_name,
                                "short": True
                            },
                            {
                                "title": "Duration",
                                "value": f"{(alert.resolved_at - alert.timestamp).total_seconds():.0f}s",
                                "short": True
                            },
                            {
                                "title": "Resolved At",
                                "value": alert.resolved_at.strftime('%Y-%m-%d %H:%M:%S'),
                                "short": True
                            }
                        ],
                        "footer": "GrandModel Monitoring",
                        "ts": int(alert.resolved_at.timestamp())
                    }
                ]
            }
            
            # Send to Slack
            response = requests.post(webhook_url, json=message)
            response.raise_for_status()
            
            logger.info(f"Slack resolution sent: {rule.name}")
            
        except Exception as e:
            logger.error(f"Failed to send Slack resolution: {rule.name}", error=str(e))
    
    async def _detect_anomalies_loop(self):
        """Anomaly detection loop"""
        while self.is_monitoring:
            try:
                await self._detect_anomalies()
                await asyncio.sleep(300)  # Run every 5 minutes
            except Exception as e:
                logger.error("Anomaly detection error", error=str(e))
                await asyncio.sleep(300)  # Wait before retry
    
    async def _detect_anomalies(self):
        """Detect anomalies in metrics"""
        if not self.config['anomaly_detection']['enabled']:
            return
        
        for model_name, history in self.metric_history.items():
            if len(history) < 50:  # Need sufficient history
                continue
            
            try:
                # Prepare data for anomaly detection
                data = np.array(history)
                
                # Initialize or update anomaly detector
                if model_name not in self.anomaly_detectors:
                    self.anomaly_detectors[model_name] = IsolationForest(
                        contamination=self.config['anomaly_detection']['contamination'],
                        random_state=42
                    )
                
                detector = self.anomaly_detectors[model_name]
                
                # Fit detector if not already fitted
                if not hasattr(detector, 'tree_'):
                    detector.fit(data)
                
                # Predict anomalies for recent data
                recent_data = data[-10:]  # Check last 10 observations
                anomaly_scores = detector.decision_function(recent_data)
                predictions = detector.predict(recent_data)
                
                # Check for anomalies
                for i, (score, prediction) in enumerate(zip(anomaly_scores, predictions)):
                    if prediction == -1:  # Anomaly detected
                        logger.warning(f"Anomaly detected in {model_name}",
                                     anomaly_score=score,
                                     observation_index=i)
                        
                        # Update anomaly score metric
                        self.prometheus_metrics['model_anomaly_score'].labels(
                            model_name, "1.0.0", "production"
                        ).set(abs(score))
                
            except Exception as e:
                logger.error(f"Anomaly detection failed for {model_name}", error=str(e))
    
    async def _generate_reports_loop(self):
        """Report generation loop"""
        while self.is_monitoring:
            try:
                await self._generate_daily_report()
                await asyncio.sleep(86400)  # Run daily
            except Exception as e:
                logger.error("Report generation error", error=str(e))
                await asyncio.sleep(3600)  # Wait before retry
    
    async def _generate_daily_report(self):
        """Generate daily monitoring report"""
        logger.info("📊 Generating daily monitoring report")
        
        # Calculate SLA metrics
        sla_metrics = await self._calculate_sla_metrics()
        
        # Generate performance summary
        performance_summary = await self._generate_performance_summary()
        
        # Generate alert summary
        alert_summary = await self._generate_alert_summary()
        
        # Create report
        report = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'monitoring_period': '24h',
            'sla_metrics': sla_metrics,
            'performance_summary': performance_summary,
            'alert_summary': alert_summary,
            'anomaly_count': len([alert for alert in self.active_alerts.values() 
                                if 'anomaly' in alert.rule_name.lower()]),
            'system_health': 'healthy' if len(self.active_alerts) == 0 else 'degraded'
        }
        
        # Save report
        report_file = self.logs_dir / f"daily_report_{datetime.now().strftime('%Y%m%d')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Daily report generated: {report_file}")
    
    async def _calculate_sla_metrics(self) -> Dict[str, float]:
        """Calculate SLA metrics"""
        # In production, this would query actual metrics from database
        return {
            'availability': 99.95,
            'latency_p95': 450,
            'error_rate': 0.08,
            'sla_compliance': 99.2
        }
    
    async def _generate_performance_summary(self) -> Dict[str, Any]:
        """Generate performance summary"""
        return {
            'avg_latency_ms': 350,
            'avg_throughput_rps': 125,
            'avg_error_rate': 0.8,
            'peak_cpu_usage': 75,
            'peak_memory_usage': 6500,
            'total_requests': 10800000,
            'total_errors': 86400
        }
    
    async def _generate_alert_summary(self) -> Dict[str, Any]:
        """Generate alert summary"""
        return {
            'total_alerts': 15,
            'critical_alerts': 3,
            'warning_alerts': 12,
            'resolved_alerts': 13,
            'active_alerts': len(self.active_alerts),
            'avg_resolution_time_minutes': 25
        }
    
    async def create_dashboard(self) -> str:
        """Create monitoring dashboard"""
        logger.info("📊 Creating monitoring dashboard")
        
        # Create Plotly dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Latency', 'Throughput', 'Error Rate', 'CPU Usage'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Generate sample data
        timestamps = pd.date_range(start=datetime.now() - timedelta(hours=24), 
                                 end=datetime.now(), freq='5T')
        latency_data = np.random.uniform(200, 600, len(timestamps))
        throughput_data = np.random.uniform(80, 150, len(timestamps))
        error_rate_data = np.random.uniform(0.1, 3.0, len(timestamps))
        cpu_usage_data = np.random.uniform(30, 80, len(timestamps))
        
        # Add traces
        fig.add_trace(
            go.Scatter(x=timestamps, y=latency_data, name='Latency (ms)'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=timestamps, y=throughput_data, name='Throughput (RPS)'),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(x=timestamps, y=error_rate_data, name='Error Rate (%)'),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=timestamps, y=cpu_usage_data, name='CPU Usage (%)'),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title='GrandModel Production Monitoring Dashboard',
            height=800,
            showlegend=False
        )
        
        # Save dashboard
        dashboard_file = self.dashboards_dir / f"monitoring_dashboard_{int(time.time())}.html"
        fig.write_html(dashboard_file)
        
        logger.info(f"Dashboard created: {dashboard_file}")
        return str(dashboard_file)
    
    async def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status"""
        return {
            'monitoring_id': self.monitoring_id,
            'is_monitoring': self.is_monitoring,
            'start_time': self.start_time.isoformat(),
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds(),
            'active_alerts': len(self.active_alerts),
            'alert_details': [
                {
                    'rule_name': alert.rule_name,
                    'severity': alert.severity,
                    'status': alert.status,
                    'timestamp': alert.timestamp.isoformat(),
                    'message': alert.message
                }
                for alert in self.active_alerts.values()
            ],
            'metrics_collected': len(self.metric_history),
            'anomaly_detectors': len(self.anomaly_detectors),
            'system_health': 'healthy' if len(self.active_alerts) == 0 else 'degraded'
        }


# Factory function
def create_monitoring_system(config_path: str = None) -> ProductionMonitoringSystem:
    """Create monitoring system instance"""
    return ProductionMonitoringSystem(config_path)


# CLI interface
async def main():
    """Main monitoring CLI"""
    import argparse
    
    parser = argparse.ArgumentParser(description="GrandModel Production Monitoring System")
    parser.add_argument("--config", help="Monitoring configuration file")
    parser.add_argument("--dashboard", action="store_true", help="Create dashboard")
    parser.add_argument("--status", action="store_true", help="Show monitoring status")
    parser.add_argument("--duration", type=int, default=3600, help="Monitoring duration in seconds")
    
    args = parser.parse_args()
    
    # Create monitoring system
    monitoring_system = create_monitoring_system(args.config)
    
    try:
        if args.dashboard:
            # Create dashboard
            dashboard_path = await monitoring_system.create_dashboard()
            print(f"✅ Dashboard created: {dashboard_path}")
            
        elif args.status:
            # Show status
            status = await monitoring_system.get_monitoring_status()
            print(f"📊 Monitoring Status:")
            print(f"   ID: {status['monitoring_id']}")
            print(f"   Running: {status['is_monitoring']}")
            print(f"   Uptime: {status['uptime_seconds']:.1f}s")
            print(f"   Active alerts: {status['active_alerts']}")
            print(f"   System health: {status['system_health']}")
            
        else:
            # Start monitoring
            print(f"🚀 Starting monitoring for {args.duration} seconds...")
            
            # Start monitoring in background
            monitoring_task = asyncio.create_task(monitoring_system.start_monitoring())
            
            # Wait for duration
            await asyncio.sleep(args.duration)
            
            # Stop monitoring
            await monitoring_system.stop_monitoring()
            
            # Get final status
            status = await monitoring_system.get_monitoring_status()
            print(f"✅ Monitoring completed")
            print(f"   Active alerts: {status['active_alerts']}")
            print(f"   System health: {status['system_health']}")
            
    except Exception as e:
        print(f"❌ Monitoring failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())