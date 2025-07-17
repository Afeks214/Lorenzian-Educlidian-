#!/usr/bin/env python3
"""
GrandModel Production Monitoring Dashboard - Agent 20 Implementation
Enterprise-grade real-time monitoring with comprehensive dashboards and alerting
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import yaml
import requests
import websocket
import pandas as pd
import numpy as np
from kubernetes import client, config
import boto3
from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram, push_to_gateway
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class MetricType(Enum):
    """Metric types"""
    GAUGE = "gauge"
    COUNTER = "counter"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

@dataclass
class MonitoringMetric:
    """Monitoring metric definition"""
    name: str
    metric_type: MetricType
    description: str
    labels: Dict[str, str] = field(default_factory=dict)
    value: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    thresholds: Dict[str, float] = field(default_factory=dict)

@dataclass
class AlertRule:
    """Alert rule definition"""
    name: str
    query: str
    condition: str  # >, <, ==, !=
    threshold: float
    severity: AlertSeverity
    duration: int  # minutes
    description: str
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)

@dataclass
class DashboardWidget:
    """Dashboard widget definition"""
    id: str
    title: str
    widget_type: str  # chart, gauge, table, stat
    query: str
    visualization_config: Dict[str, Any] = field(default_factory=dict)
    position: Dict[str, int] = field(default_factory=dict)
    refresh_interval: int = 30  # seconds

@dataclass
class Dashboard:
    """Dashboard definition"""
    id: str
    title: str
    description: str
    widgets: List[DashboardWidget] = field(default_factory=list)
    refresh_interval: int = 30
    tags: List[str] = field(default_factory=list)

class PrometheusClient:
    """Prometheus client for metrics collection"""
    
    def __init__(self, prometheus_url: str):
        self.prometheus_url = prometheus_url
        self.session = requests.Session()
    
    async def query(self, query: str) -> Dict[str, Any]:
        """Execute Prometheus query"""
        try:
            response = self.session.get(
                f"{self.prometheus_url}/api/v1/query",
                params={"query": query},
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Prometheus query failed: {response.status_code}")
                return {}
        except Exception as e:
            logger.error(f"Error querying Prometheus: {e}")
            return {}
    
    async def query_range(self, query: str, start: datetime, end: datetime, step: str = "1m") -> Dict[str, Any]:
        """Execute Prometheus range query"""
        try:
            response = self.session.get(
                f"{self.prometheus_url}/api/v1/query_range",
                params={
                    "query": query,
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "step": step
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Prometheus range query failed: {response.status_code}")
                return {}
        except Exception as e:
            logger.error(f"Error querying Prometheus range: {e}")
            return {}
    
    async def get_metrics(self) -> List[str]:
        """Get available metrics"""
        try:
            response = self.session.get(
                f"{self.prometheus_url}/api/v1/label/__name__/values",
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get('data', [])
            return []
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return []

class MetricsCollector:
    """Collect and process metrics from various sources"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.prometheus_client = PrometheusClient(config['prometheus_url'])
        self.metrics_cache = {}
        self.last_collection_time = {}
        
        # Load Kubernetes config
        try:
            config.load_incluster_config()
        except:
            config.load_kube_config()
    
    async def collect_system_metrics(self) -> Dict[str, MonitoringMetric]:
        """Collect system-level metrics"""
        metrics = {}
        
        try:
            # CPU utilization
            cpu_query = 'rate(container_cpu_usage_seconds_total[5m])'
            cpu_result = await self.prometheus_client.query(cpu_query)
            
            if cpu_result and cpu_result['data']['result']:
                for result in cpu_result['data']['result']:
                    pod_name = result['metric'].get('pod', 'unknown')
                    cpu_value = float(result['value'][1])
                    
                    metrics[f"cpu_usage_{pod_name}"] = MonitoringMetric(
                        name=f"cpu_usage_{pod_name}",
                        metric_type=MetricType.GAUGE,
                        description=f"CPU usage for pod {pod_name}",
                        labels=result['metric'],
                        value=cpu_value,
                        thresholds={"warning": 0.7, "critical": 0.9}
                    )
            
            # Memory utilization
            memory_query = 'container_memory_usage_bytes / container_spec_memory_limit_bytes'
            memory_result = await self.prometheus_client.query(memory_query)
            
            if memory_result and memory_result['data']['result']:
                for result in memory_result['data']['result']:
                    pod_name = result['metric'].get('pod', 'unknown')
                    memory_value = float(result['value'][1])
                    
                    metrics[f"memory_usage_{pod_name}"] = MonitoringMetric(
                        name=f"memory_usage_{pod_name}",
                        metric_type=MetricType.GAUGE,
                        description=f"Memory usage for pod {pod_name}",
                        labels=result['metric'],
                        value=memory_value,
                        thresholds={"warning": 0.8, "critical": 0.95}
                    )
            
            # Disk usage
            disk_query = 'filesystem_avail_bytes / filesystem_size_bytes'
            disk_result = await self.prometheus_client.query(disk_query)
            
            if disk_result and disk_result['data']['result']:
                for result in disk_result['data']['result']:
                    device = result['metric'].get('device', 'unknown')
                    disk_value = 1 - float(result['value'][1])  # Used space
                    
                    metrics[f"disk_usage_{device}"] = MonitoringMetric(
                        name=f"disk_usage_{device}",
                        metric_type=MetricType.GAUGE,
                        description=f"Disk usage for device {device}",
                        labels=result['metric'],
                        value=disk_value,
                        thresholds={"warning": 0.8, "critical": 0.9}
                    )
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
        
        return metrics
    
    async def collect_application_metrics(self) -> Dict[str, MonitoringMetric]:
        """Collect application-specific metrics"""
        metrics = {}
        
        try:
            # Request rate
            request_rate_query = 'rate(http_requests_total[5m])'
            request_rate_result = await self.prometheus_client.query(request_rate_query)
            
            if request_rate_result and request_rate_result['data']['result']:
                for result in request_rate_result['data']['result']:
                    service = result['metric'].get('job', 'unknown')
                    rate_value = float(result['value'][1])
                    
                    metrics[f"request_rate_{service}"] = MonitoringMetric(
                        name=f"request_rate_{service}",
                        metric_type=MetricType.GAUGE,
                        description=f"Request rate for service {service}",
                        labels=result['metric'],
                        value=rate_value,
                        thresholds={"warning": 1000, "critical": 5000}
                    )
            
            # Error rate
            error_rate_query = 'rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])'
            error_rate_result = await self.prometheus_client.query(error_rate_query)
            
            if error_rate_result and error_rate_result['data']['result']:
                for result in error_rate_result['data']['result']:
                    service = result['metric'].get('job', 'unknown')
                    error_rate_value = float(result['value'][1])
                    
                    metrics[f"error_rate_{service}"] = MonitoringMetric(
                        name=f"error_rate_{service}",
                        metric_type=MetricType.GAUGE,
                        description=f"Error rate for service {service}",
                        labels=result['metric'],
                        value=error_rate_value,
                        thresholds={"warning": 0.01, "critical": 0.05}
                    )
            
            # Response time
            response_time_query = 'histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))'
            response_time_result = await self.prometheus_client.query(response_time_query)
            
            if response_time_result and response_time_result['data']['result']:
                for result in response_time_result['data']['result']:
                    service = result['metric'].get('job', 'unknown')
                    response_time_value = float(result['value'][1])
                    
                    metrics[f"response_time_p95_{service}"] = MonitoringMetric(
                        name=f"response_time_p95_{service}",
                        metric_type=MetricType.GAUGE,
                        description=f"95th percentile response time for service {service}",
                        labels=result['metric'],
                        value=response_time_value,
                        thresholds={"warning": 0.1, "critical": 0.5}
                    )
            
        except Exception as e:
            logger.error(f"Error collecting application metrics: {e}")
        
        return metrics
    
    async def collect_business_metrics(self) -> Dict[str, MonitoringMetric]:
        """Collect business-specific metrics"""
        metrics = {}
        
        try:
            # Strategic agent metrics
            strategic_latency_query = 'histogram_quantile(0.95, strategic_latency_seconds_bucket)'
            strategic_result = await self.prometheus_client.query(strategic_latency_query)
            
            if strategic_result and strategic_result['data']['result']:
                for result in strategic_result['data']['result']:
                    latency_value = float(result['value'][1])
                    
                    metrics["strategic_latency_p95"] = MonitoringMetric(
                        name="strategic_latency_p95",
                        metric_type=MetricType.GAUGE,
                        description="Strategic agent 95th percentile latency",
                        labels=result['metric'],
                        value=latency_value,
                        thresholds={"warning": 0.002, "critical": 0.005}
                    )
            
            # Tactical agent metrics
            tactical_throughput_query = 'tactical_throughput_rps'
            tactical_result = await self.prometheus_client.query(tactical_throughput_query)
            
            if tactical_result and tactical_result['data']['result']:
                for result in tactical_result['data']['result']:
                    throughput_value = float(result['value'][1])
                    
                    metrics["tactical_throughput"] = MonitoringMetric(
                        name="tactical_throughput",
                        metric_type=MetricType.GAUGE,
                        description="Tactical agent throughput (requests per second)",
                        labels=result['metric'],
                        value=throughput_value,
                        thresholds={"warning": 100, "critical": 50}
                    )
            
            # Risk management metrics
            var_calculation_query = 'histogram_quantile(0.95, risk_var_calculation_seconds_bucket)'
            var_result = await self.prometheus_client.query(var_calculation_query)
            
            if var_result and var_result['data']['result']:
                for result in var_result['data']['result']:
                    var_time_value = float(result['value'][1])
                    
                    metrics["var_calculation_time_p95"] = MonitoringMetric(
                        name="var_calculation_time_p95",
                        metric_type=MetricType.GAUGE,
                        description="VaR calculation 95th percentile time",
                        labels=result['metric'],
                        value=var_time_value,
                        thresholds={"warning": 0.005, "critical": 0.01}
                    )
            
            # Correlation shock alerts
            correlation_shock_query = 'risk_correlation_shock_alert'
            correlation_result = await self.prometheus_client.query(correlation_shock_query)
            
            if correlation_result and correlation_result['data']['result']:
                for result in correlation_result['data']['result']:
                    shock_value = float(result['value'][1])
                    
                    metrics["correlation_shock_alert"] = MonitoringMetric(
                        name="correlation_shock_alert",
                        metric_type=MetricType.GAUGE,
                        description="Correlation shock alert status",
                        labels=result['metric'],
                        value=shock_value,
                        thresholds={"warning": 0.5, "critical": 1.0}
                    )
            
        except Exception as e:
            logger.error(f"Error collecting business metrics: {e}")
        
        return metrics
    
    async def collect_all_metrics(self) -> Dict[str, MonitoringMetric]:
        """Collect all metrics"""
        all_metrics = {}
        
        # Collect from all sources
        system_metrics = await self.collect_system_metrics()
        application_metrics = await self.collect_application_metrics()
        business_metrics = await self.collect_business_metrics()
        
        # Combine all metrics
        all_metrics.update(system_metrics)
        all_metrics.update(application_metrics)
        all_metrics.update(business_metrics)
        
        # Cache metrics
        self.metrics_cache = all_metrics
        self.last_collection_time = datetime.now()
        
        return all_metrics

class AlertManager:
    """Alert management system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.alert_rules = self.load_alert_rules()
        self.active_alerts = {}
        self.alert_history = []
        
    def load_alert_rules(self) -> List[AlertRule]:
        """Load alert rules from configuration"""
        rules = []
        
        # System alert rules
        system_rules = [
            AlertRule(
                name="high_cpu_usage",
                query="rate(container_cpu_usage_seconds_total[5m])",
                condition=">",
                threshold=0.8,
                severity=AlertSeverity.WARNING,
                duration=5,
                description="High CPU usage detected",
                labels={"team": "infrastructure"},
                annotations={"runbook_url": "https://runbook.grandmodel.com/high-cpu"}
            ),
            AlertRule(
                name="high_memory_usage",
                query="container_memory_usage_bytes / container_spec_memory_limit_bytes",
                condition=">",
                threshold=0.9,
                severity=AlertSeverity.CRITICAL,
                duration=2,
                description="High memory usage detected",
                labels={"team": "infrastructure"},
                annotations={"runbook_url": "https://runbook.grandmodel.com/high-memory"}
            ),
            AlertRule(
                name="service_down",
                query="up{job=~'grandmodel.*'}",
                condition="==",
                threshold=0,
                severity=AlertSeverity.CRITICAL,
                duration=1,
                description="Service is down",
                labels={"team": "engineering"},
                annotations={"runbook_url": "https://runbook.grandmodel.com/service-down"}
            )
        ]
        
        # Application alert rules
        application_rules = [
            AlertRule(
                name="high_error_rate",
                query="rate(http_requests_total{status=~'5..'}[5m]) / rate(http_requests_total[5m])",
                condition=">",
                threshold=0.05,
                severity=AlertSeverity.CRITICAL,
                duration=3,
                description="High error rate detected",
                labels={"team": "engineering"},
                annotations={"runbook_url": "https://runbook.grandmodel.com/high-error-rate"}
            ),
            AlertRule(
                name="high_response_time",
                query="histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
                condition=">",
                threshold=0.5,
                severity=AlertSeverity.WARNING,
                duration=5,
                description="High response time detected",
                labels={"team": "engineering"},
                annotations={"runbook_url": "https://runbook.grandmodel.com/high-latency"}
            )
        ]
        
        # Business alert rules
        business_rules = [
            AlertRule(
                name="strategic_latency_high",
                query="histogram_quantile(0.95, strategic_latency_seconds_bucket)",
                condition=">",
                threshold=0.002,
                severity=AlertSeverity.WARNING,
                duration=2,
                description="Strategic agent latency is high",
                labels={"team": "trading"},
                annotations={"runbook_url": "https://runbook.grandmodel.com/strategic-latency"}
            ),
            AlertRule(
                name="var_calculation_slow",
                query="histogram_quantile(0.95, risk_var_calculation_seconds_bucket)",
                condition=">",
                threshold=0.005,
                severity=AlertSeverity.WARNING,
                duration=3,
                description="VaR calculation is slow",
                labels={"team": "risk"},
                annotations={"runbook_url": "https://runbook.grandmodel.com/var-slow"}
            ),
            AlertRule(
                name="correlation_shock",
                query="risk_correlation_shock_alert",
                condition="==",
                threshold=1,
                severity=AlertSeverity.CRITICAL,
                duration=0,
                description="Correlation shock detected",
                labels={"team": "risk"},
                annotations={"runbook_url": "https://runbook.grandmodel.com/correlation-shock"}
            )
        ]
        
        rules.extend(system_rules)
        rules.extend(application_rules)
        rules.extend(business_rules)
        
        return rules
    
    async def evaluate_alerts(self, metrics: Dict[str, MonitoringMetric]) -> List[Dict[str, Any]]:
        """Evaluate alert rules against current metrics"""
        alerts = []
        
        for rule in self.alert_rules:
            try:
                # Find matching metrics
                matching_metrics = [
                    metric for metric in metrics.values()
                    if self.metric_matches_rule(metric, rule)
                ]
                
                for metric in matching_metrics:
                    if self.evaluate_condition(metric.value, rule.condition, rule.threshold):
                        alert = {
                            'rule_name': rule.name,
                            'metric_name': metric.name,
                            'severity': rule.severity.value,
                            'description': rule.description,
                            'value': metric.value,
                            'threshold': rule.threshold,
                            'labels': {**rule.labels, **metric.labels},
                            'annotations': rule.annotations,
                            'timestamp': datetime.now().isoformat()
                        }
                        alerts.append(alert)
                        
                        # Track active alerts
                        alert_key = f"{rule.name}_{metric.name}"
                        self.active_alerts[alert_key] = alert
                        
            except Exception as e:
                logger.error(f"Error evaluating alert rule {rule.name}: {e}")
        
        return alerts
    
    def metric_matches_rule(self, metric: MonitoringMetric, rule: AlertRule) -> bool:
        """Check if metric matches alert rule"""
        # This is a simplified matching logic
        # In practice, you'd use more sophisticated PromQL parsing
        return rule.name.replace('_', '') in metric.name.replace('_', '').lower()
    
    def evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Evaluate alert condition"""
        if condition == ">":
            return value > threshold
        elif condition == "<":
            return value < threshold
        elif condition == "==":
            return value == threshold
        elif condition == "!=":
            return value != threshold
        elif condition == ">=":
            return value >= threshold
        elif condition == "<=":
            return value <= threshold
        else:
            return False

class DashboardManager:
    """Dashboard management system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dashboards = self.create_default_dashboards()
        self.prometheus_client = PrometheusClient(config['prometheus_url'])
    
    def create_default_dashboards(self) -> Dict[str, Dashboard]:
        """Create default dashboards"""
        dashboards = {}
        
        # System Overview Dashboard
        system_dashboard = Dashboard(
            id="system_overview",
            title="System Overview",
            description="High-level system health and performance metrics",
            tags=["system", "infrastructure"]
        )
        
        system_dashboard.widgets = [
            DashboardWidget(
                id="cpu_usage",
                title="CPU Usage",
                widget_type="gauge",
                query="rate(container_cpu_usage_seconds_total[5m])",
                visualization_config={
                    "min": 0,
                    "max": 1,
                    "thresholds": [0.7, 0.9]
                },
                position={"x": 0, "y": 0, "w": 6, "h": 4}
            ),
            DashboardWidget(
                id="memory_usage",
                title="Memory Usage",
                widget_type="gauge",
                query="container_memory_usage_bytes / container_spec_memory_limit_bytes",
                visualization_config={
                    "min": 0,
                    "max": 1,
                    "thresholds": [0.8, 0.95]
                },
                position={"x": 6, "y": 0, "w": 6, "h": 4}
            ),
            DashboardWidget(
                id="request_rate",
                title="Request Rate",
                widget_type="chart",
                query="rate(http_requests_total[5m])",
                visualization_config={
                    "chart_type": "line",
                    "time_range": "1h"
                },
                position={"x": 0, "y": 4, "w": 12, "h": 6}
            ),
            DashboardWidget(
                id="error_rate",
                title="Error Rate",
                widget_type="chart",
                query="rate(http_requests_total{status=~'5..'}[5m]) / rate(http_requests_total[5m])",
                visualization_config={
                    "chart_type": "line",
                    "time_range": "1h",
                    "color": "red"
                },
                position={"x": 0, "y": 10, "w": 12, "h": 6}
            )
        ]
        
        dashboards["system_overview"] = system_dashboard
        
        # Trading Dashboard
        trading_dashboard = Dashboard(
            id="trading_performance",
            title="Trading Performance",
            description="Trading system performance and business metrics",
            tags=["trading", "business"]
        )
        
        trading_dashboard.widgets = [
            DashboardWidget(
                id="strategic_latency",
                title="Strategic Agent Latency",
                widget_type="gauge",
                query="histogram_quantile(0.95, strategic_latency_seconds_bucket)",
                visualization_config={
                    "min": 0,
                    "max": 0.01,
                    "thresholds": [0.002, 0.005],
                    "unit": "seconds"
                },
                position={"x": 0, "y": 0, "w": 4, "h": 4}
            ),
            DashboardWidget(
                id="tactical_throughput",
                title="Tactical Agent Throughput",
                widget_type="gauge",
                query="tactical_throughput_rps",
                visualization_config={
                    "min": 0,
                    "max": 1000,
                    "thresholds": [100, 50],
                    "unit": "rps"
                },
                position={"x": 4, "y": 0, "w": 4, "h": 4}
            ),
            DashboardWidget(
                id="var_calculation_time",
                title="VaR Calculation Time",
                widget_type="gauge",
                query="histogram_quantile(0.95, risk_var_calculation_seconds_bucket)",
                visualization_config={
                    "min": 0,
                    "max": 0.01,
                    "thresholds": [0.005, 0.01],
                    "unit": "seconds"
                },
                position={"x": 8, "y": 0, "w": 4, "h": 4}
            ),
            DashboardWidget(
                id="correlation_matrix",
                title="Correlation Matrix",
                widget_type="heatmap",
                query="risk_correlation_matrix",
                visualization_config={
                    "colorscale": "RdBu",
                    "min": -1,
                    "max": 1
                },
                position={"x": 0, "y": 4, "w": 12, "h": 8}
            )
        ]
        
        dashboards["trading_performance"] = trading_dashboard
        
        # Risk Dashboard
        risk_dashboard = Dashboard(
            id="risk_management",
            title="Risk Management",
            description="Risk metrics and compliance monitoring",
            tags=["risk", "compliance"]
        )
        
        risk_dashboard.widgets = [
            DashboardWidget(
                id="current_var",
                title="Current VaR",
                widget_type="stat",
                query="risk_var_current",
                visualization_config={
                    "color": "blue",
                    "unit": "%"
                },
                position={"x": 0, "y": 0, "w": 3, "h": 3}
            ),
            DashboardWidget(
                id="var_limit",
                title="VaR Limit",
                widget_type="stat",
                query="risk_var_limit",
                visualization_config={
                    "color": "red",
                    "unit": "%"
                },
                position={"x": 3, "y": 0, "w": 3, "h": 3}
            ),
            DashboardWidget(
                id="var_utilization",
                title="VaR Utilization",
                widget_type="gauge",
                query="risk_var_current / risk_var_limit",
                visualization_config={
                    "min": 0,
                    "max": 1,
                    "thresholds": [0.8, 0.9],
                    "unit": "%"
                },
                position={"x": 6, "y": 0, "w": 6, "h": 3}
            ),
            DashboardWidget(
                id="correlation_shock_alerts",
                title="Correlation Shock Alerts",
                widget_type="table",
                query="risk_correlation_shock_alert",
                visualization_config={
                    "columns": ["timestamp", "severity", "description"]
                },
                position={"x": 0, "y": 3, "w": 12, "h": 6}
            )
        ]
        
        dashboards["risk_management"] = risk_dashboard
        
        return dashboards
    
    async def get_dashboard_data(self, dashboard_id: str) -> Dict[str, Any]:
        """Get dashboard data"""
        dashboard = self.dashboards.get(dashboard_id)
        if not dashboard:
            return {}
        
        dashboard_data = {
            "id": dashboard.id,
            "title": dashboard.title,
            "description": dashboard.description,
            "widgets": []
        }
        
        for widget in dashboard.widgets:
            widget_data = await self.get_widget_data(widget)
            dashboard_data["widgets"].append(widget_data)
        
        return dashboard_data
    
    async def get_widget_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Get widget data"""
        widget_data = {
            "id": widget.id,
            "title": widget.title,
            "type": widget.widget_type,
            "position": widget.position,
            "config": widget.visualization_config,
            "data": None,
            "last_updated": datetime.now().isoformat()
        }
        
        try:
            # Query data from Prometheus
            query_result = await self.prometheus_client.query(widget.query)
            
            if query_result and query_result.get('data', {}).get('result'):
                # Process data based on widget type
                if widget.widget_type == "gauge":
                    widget_data["data"] = self.process_gauge_data(query_result['data']['result'])
                elif widget.widget_type == "chart":
                    widget_data["data"] = self.process_chart_data(query_result['data']['result'])
                elif widget.widget_type == "stat":
                    widget_data["data"] = self.process_stat_data(query_result['data']['result'])
                elif widget.widget_type == "table":
                    widget_data["data"] = self.process_table_data(query_result['data']['result'])
                elif widget.widget_type == "heatmap":
                    widget_data["data"] = self.process_heatmap_data(query_result['data']['result'])
                else:
                    widget_data["data"] = query_result['data']['result']
        
        except Exception as e:
            logger.error(f"Error getting widget data for {widget.id}: {e}")
            widget_data["error"] = str(e)
        
        return widget_data
    
    def process_gauge_data(self, result: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process data for gauge widget"""
        if not result:
            return {"value": 0}
        
        # Take the first result
        first_result = result[0]
        value = float(first_result['value'][1])
        
        return {
            "value": value,
            "labels": first_result['metric']
        }
    
    def process_chart_data(self, result: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process data for chart widget"""
        if not result:
            return {"series": []}
        
        series = []
        for item in result:
            series.append({
                "name": item['metric'].get('job', 'unknown'),
                "data": [[int(item['value'][0]) * 1000, float(item['value'][1])]],
                "labels": item['metric']
            })
        
        return {"series": series}
    
    def process_stat_data(self, result: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process data for stat widget"""
        if not result:
            return {"value": 0}
        
        first_result = result[0]
        value = float(first_result['value'][1])
        
        return {
            "value": value,
            "labels": first_result['metric']
        }
    
    def process_table_data(self, result: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process data for table widget"""
        if not result:
            return {"rows": []}
        
        rows = []
        for item in result:
            row = {
                "timestamp": datetime.fromtimestamp(int(item['value'][0])).isoformat(),
                "value": float(item['value'][1]),
                "labels": item['metric']
            }
            rows.append(row)
        
        return {"rows": rows}
    
    def process_heatmap_data(self, result: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process data for heatmap widget"""
        if not result:
            return {"data": []}
        
        # This would process correlation matrix data
        # For now, return a simple structure
        return {
            "data": [[i, j, np.random.random() * 2 - 1] for i in range(5) for j in range(5)]
        }

class RealTimeDashboard:
    """Real-time dashboard web application"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_collector = MetricsCollector(config)
        self.alert_manager = AlertManager(config)
        self.dashboard_manager = DashboardManager(config)
        
        # Create Flask app
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = config.get('secret_key', 'dev-secret-key')
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Set up routes
        self.setup_routes()
        self.setup_websocket_handlers()
        
        # Background task for real-time updates
        self.update_interval = config.get('update_interval', 30)
        self.running = False
    
    def setup_routes(self):
        """Set up Flask routes"""
        
        @self.app.route('/')
        def index():
            return render_template('dashboard.html')
        
        @self.app.route('/api/dashboards')
        def get_dashboards():
            return jsonify(list(self.dashboard_manager.dashboards.keys()))
        
        @self.app.route('/api/dashboards/<dashboard_id>')
        async def get_dashboard(dashboard_id):
            dashboard_data = await self.dashboard_manager.get_dashboard_data(dashboard_id)
            return jsonify(dashboard_data)
        
        @self.app.route('/api/metrics')
        async def get_metrics():
            metrics = await self.metrics_collector.collect_all_metrics()
            return jsonify({
                name: {
                    'value': metric.value,
                    'timestamp': metric.timestamp.isoformat(),
                    'labels': metric.labels,
                    'thresholds': metric.thresholds
                }
                for name, metric in metrics.items()
            })
        
        @self.app.route('/api/alerts')
        async def get_alerts():
            metrics = await self.metrics_collector.collect_all_metrics()
            alerts = await self.alert_manager.evaluate_alerts(metrics)
            return jsonify(alerts)
        
        @self.app.route('/api/health')
        def health_check():
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'version': '1.0.0'
            })
    
    def setup_websocket_handlers(self):
        """Set up WebSocket handlers"""
        
        @self.socketio.on('connect')
        def handle_connect():
            logger.info(f"Client connected: {request.sid}")
            emit('connected', {'message': 'Connected to GrandModel monitoring'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            logger.info(f"Client disconnected: {request.sid}")
        
        @self.socketio.on('subscribe_dashboard')
        def handle_subscribe_dashboard(data):
            dashboard_id = data.get('dashboard_id')
            logger.info(f"Client {request.sid} subscribed to dashboard {dashboard_id}")
            # Add client to dashboard room
            # join_room(dashboard_id)
        
        @self.socketio.on('unsubscribe_dashboard')
        def handle_unsubscribe_dashboard(data):
            dashboard_id = data.get('dashboard_id')
            logger.info(f"Client {request.sid} unsubscribed from dashboard {dashboard_id}")
            # Remove client from dashboard room
            # leave_room(dashboard_id)
    
    async def start_real_time_updates(self):
        """Start real-time updates"""
        self.running = True
        
        while self.running:
            try:
                # Collect metrics
                metrics = await self.metrics_collector.collect_all_metrics()
                
                # Evaluate alerts
                alerts = await self.alert_manager.evaluate_alerts(metrics)
                
                # Emit updates to connected clients
                self.socketio.emit('metrics_update', {
                    'timestamp': datetime.now().isoformat(),
                    'metrics': {
                        name: {
                            'value': metric.value,
                            'timestamp': metric.timestamp.isoformat(),
                            'labels': metric.labels
                        }
                        for name, metric in metrics.items()
                    }
                })
                
                if alerts:
                    self.socketio.emit('alerts_update', {
                        'timestamp': datetime.now().isoformat(),
                        'alerts': alerts
                    })
                
                # Wait before next update
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in real-time updates: {e}")
                await asyncio.sleep(self.update_interval)
    
    def run(self, host: str = '0.0.0.0', port: int = 8080, debug: bool = False):
        """Run the dashboard application"""
        logger.info(f"Starting GrandModel monitoring dashboard on {host}:{port}")
        
        # Start real-time updates in background
        asyncio.create_task(self.start_real_time_updates())
        
        # Run Flask app
        self.socketio.run(self.app, host=host, port=port, debug=debug)
    
    def stop(self):
        """Stop the dashboard application"""
        self.running = False
        logger.info("Stopping GrandModel monitoring dashboard")

# Dashboard HTML template
DASHBOARD_HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>GrandModel Production Dashboard</title>
    <script src="https://cdn.socket.io/4.0.0/socket.io.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 5px;
        }
        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }
        .widget {
            background-color: white;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .widget-title {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 15px;
            color: #2c3e50;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #27ae60;
        }
        .metric-warning {
            color: #f39c12;
        }
        .metric-critical {
            color: #e74c3c;
        }
        .status-indicator {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 10px;
        }
        .status-healthy {
            background-color: #27ae60;
        }
        .status-warning {
            background-color: #f39c12;
        }
        .status-critical {
            background-color: #e74c3c;
        }
        .alert-banner {
            background-color: #e74c3c;
            color: white;
            padding: 10px;
            margin-bottom: 20px;
            border-radius: 5px;
            display: none;
        }
        .last-updated {
            font-size: 12px;
            color: #7f8c8d;
            margin-top: 10px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 GrandModel Production Dashboard</h1>
        <p>Real-time monitoring and alerting for enterprise trading systems</p>
    </div>
    
    <div id="alert-banner" class="alert-banner">
        <strong>Alert:</strong> <span id="alert-message"></span>
    </div>
    
    <div class="dashboard-grid">
        <div class="widget">
            <div class="widget-title">System Health</div>
            <div id="system-health">
                <div><span class="status-indicator status-healthy"></span>Strategic Agent: <span id="strategic-status">Healthy</span></div>
                <div><span class="status-indicator status-healthy"></span>Tactical Agent: <span id="tactical-status">Healthy</span></div>
                <div><span class="status-indicator status-healthy"></span>Risk Management: <span id="risk-status">Healthy</span></div>
            </div>
            <div class="last-updated">Last updated: <span id="health-timestamp">--</span></div>
        </div>
        
        <div class="widget">
            <div class="widget-title">Performance Metrics</div>
            <div>
                <div>Strategic Latency: <span id="strategic-latency" class="metric-value">-- ms</span></div>
                <div>Tactical Throughput: <span id="tactical-throughput" class="metric-value">-- rps</span></div>
                <div>VaR Calculation: <span id="var-time" class="metric-value">-- ms</span></div>
            </div>
            <div class="last-updated">Last updated: <span id="performance-timestamp">--</span></div>
        </div>
        
        <div class="widget">
            <div class="widget-title">Risk Metrics</div>
            <div>
                <div>Current VaR: <span id="current-var" class="metric-value">--%</span></div>
                <div>VaR Utilization: <span id="var-utilization" class="metric-value">--%</span></div>
                <div>Correlation Shocks: <span id="correlation-shocks" class="metric-value">--</span></div>
            </div>
            <div class="last-updated">Last updated: <span id="risk-timestamp">--</span></div>
        </div>
        
        <div class="widget">
            <div class="widget-title">System Resources</div>
            <div>
                <div>CPU Usage: <span id="cpu-usage" class="metric-value">--%</span></div>
                <div>Memory Usage: <span id="memory-usage" class="metric-value">--%</span></div>
                <div>Request Rate: <span id="request-rate" class="metric-value">-- rps</span></div>
            </div>
            <div class="last-updated">Last updated: <span id="resources-timestamp">--</span></div>
        </div>
        
        <div class="widget">
            <div class="widget-title">Request Rate</div>
            <div id="request-rate-chart"></div>
            <div class="last-updated">Last updated: <span id="chart-timestamp">--</span></div>
        </div>
        
        <div class="widget">
            <div class="widget-title">Active Alerts</div>
            <div id="active-alerts">No active alerts</div>
            <div class="last-updated">Last updated: <span id="alerts-timestamp">--</span></div>
        </div>
    </div>
    
    <script>
        const socket = io();
        
        socket.on('connect', function() {
            console.log('Connected to monitoring server');
        });
        
        socket.on('metrics_update', function(data) {
            updateMetrics(data.metrics);
            updateTimestamp('health-timestamp', data.timestamp);
            updateTimestamp('performance-timestamp', data.timestamp);
            updateTimestamp('risk-timestamp', data.timestamp);
            updateTimestamp('resources-timestamp', data.timestamp);
            updateTimestamp('chart-timestamp', data.timestamp);
        });
        
        socket.on('alerts_update', function(data) {
            updateAlerts(data.alerts);
            updateTimestamp('alerts-timestamp', data.timestamp);
        });
        
        function updateMetrics(metrics) {
            // Update strategic metrics
            if (metrics.strategic_latency_p95) {
                document.getElementById('strategic-latency').textContent = (metrics.strategic_latency_p95.value * 1000).toFixed(2) + ' ms';
                updateMetricColor('strategic-latency', metrics.strategic_latency_p95.value, 0.002, 0.005);
            }
            
            // Update tactical metrics
            if (metrics.tactical_throughput) {
                document.getElementById('tactical-throughput').textContent = Math.round(metrics.tactical_throughput.value) + ' rps';
                updateMetricColor('tactical-throughput', metrics.tactical_throughput.value, 100, 50);
            }
            
            // Update VaR metrics
            if (metrics.var_calculation_time_p95) {
                document.getElementById('var-time').textContent = (metrics.var_calculation_time_p95.value * 1000).toFixed(2) + ' ms';
                updateMetricColor('var-time', metrics.var_calculation_time_p95.value, 0.005, 0.01);
            }
            
            // Update system resources
            updateSystemResources(metrics);
        }
        
        function updateSystemResources(metrics) {
            // Find CPU metrics
            for (const [key, metric] of Object.entries(metrics)) {
                if (key.includes('cpu_usage')) {
                    document.getElementById('cpu-usage').textContent = (metric.value * 100).toFixed(1) + '%';
                    updateMetricColor('cpu-usage', metric.value, 0.7, 0.9);
                    break;
                }
            }
            
            // Find memory metrics
            for (const [key, metric] of Object.entries(metrics)) {
                if (key.includes('memory_usage')) {
                    document.getElementById('memory-usage').textContent = (metric.value * 100).toFixed(1) + '%';
                    updateMetricColor('memory-usage', metric.value, 0.8, 0.95);
                    break;
                }
            }
            
            // Find request rate metrics
            for (const [key, metric] of Object.entries(metrics)) {
                if (key.includes('request_rate')) {
                    document.getElementById('request-rate').textContent = Math.round(metric.value) + ' rps';
                    break;
                }
            }
        }
        
        function updateMetricColor(elementId, value, warningThreshold, criticalThreshold) {
            const element = document.getElementById(elementId);
            element.className = 'metric-value';
            
            if (value >= criticalThreshold) {
                element.classList.add('metric-critical');
            } else if (value >= warningThreshold) {
                element.classList.add('metric-warning');
            }
        }
        
        function updateAlerts(alerts) {
            const alertsContainer = document.getElementById('active-alerts');
            const alertBanner = document.getElementById('alert-banner');
            const alertMessage = document.getElementById('alert-message');
            
            if (alerts.length === 0) {
                alertsContainer.textContent = 'No active alerts';
                alertBanner.style.display = 'none';
            } else {
                alertsContainer.innerHTML = '';
                let criticalAlerts = alerts.filter(a => a.severity === 'critical');
                
                if (criticalAlerts.length > 0) {
                    alertBanner.style.display = 'block';
                    alertMessage.textContent = `${criticalAlerts.length} critical alert(s) active`;
                }
                
                alerts.forEach(alert => {
                    const alertDiv = document.createElement('div');
                    alertDiv.innerHTML = `
                        <strong>${alert.severity.toUpperCase()}:</strong> ${alert.description}
                        <br><small>${alert.metric_name}: ${alert.value}</small>
                    `;
                    alertDiv.style.margin = '10px 0';
                    alertDiv.style.padding = '10px';
                    alertDiv.style.borderRadius = '3px';
                    alertDiv.style.backgroundColor = alert.severity === 'critical' ? '#fee' : '#ffd';
                    alertsContainer.appendChild(alertDiv);
                });
            }
        }
        
        function updateTimestamp(elementId, timestamp) {
            const element = document.getElementById(elementId);
            if (element) {
                element.textContent = new Date(timestamp).toLocaleTimeString();
            }
        }
        
        // Initialize request rate chart
        const requestRateChart = {
            data: [{
                x: [],
                y: [],
                type: 'scatter',
                mode: 'lines',
                name: 'Request Rate'
            }],
            layout: {
                title: '',
                xaxis: { title: 'Time' },
                yaxis: { title: 'Requests/sec' },
                margin: { t: 20, b: 40, l: 60, r: 20 }
            }
        };
        
        Plotly.newPlot('request-rate-chart', requestRateChart.data, requestRateChart.layout);
        
        // Request initial data
        fetch('/api/metrics').then(response => response.json()).then(data => {
            updateMetrics(data);
        });
        
        fetch('/api/alerts').then(response => response.json()).then(data => {
            updateAlerts(data);
        });
    </script>
</body>
</html>
"""

# Create templates directory and file
import os
def create_dashboard_template():
    """Create dashboard HTML template"""
    templates_dir = "/app/templates"
    os.makedirs(templates_dir, exist_ok=True)
    
    with open(f"{templates_dir}/dashboard.html", "w") as f:
        f.write(DASHBOARD_HTML_TEMPLATE)

# Example usage and testing
async def main():
    """Main function for testing"""
    # Create templates
    create_dashboard_template()
    
    # Configuration
    config = {
        'prometheus_url': 'http://prometheus:9090',
        'secret_key': 'production-secret-key',
        'update_interval': 30,
        'aws_region': 'us-east-1'
    }
    
    # Initialize dashboard
    dashboard = RealTimeDashboard(config)
    
    # Run dashboard
    dashboard.run(host='0.0.0.0', port=8080, debug=False)

if __name__ == "__main__":
    asyncio.run(main())