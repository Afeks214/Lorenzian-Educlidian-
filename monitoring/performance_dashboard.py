#!/usr/bin/env python3
"""
Performance Dashboard for GrandModel MARL Trading System
Real-time visualization and monitoring dashboard with interactive components
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor

# Web framework
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

# Visualization
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.utils

# Monitoring imports
from .health_check_system import ComprehensiveHealthCheckSystem
from .superposition_monitoring import SuperpositionMonitor
from .alerting_system import AlertManager
from .diagnostic_tools import SuperpositionDiagnosticSuite

# Metrics
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import redis
import psutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dashboard metrics
DASHBOARD_VIEWS = Counter('dashboard_views_total', 'Total dashboard views', ['dashboard_type'])
DASHBOARD_RESPONSE_TIME = Histogram('dashboard_response_time_seconds', 'Dashboard response time', ['endpoint'])
ACTIVE_WEBSOCKETS = Gauge('active_websockets', 'Number of active WebSocket connections')

class DashboardType(Enum):
    """Dashboard types."""
    OVERVIEW = "overview"
    SUPERPOSITION = "superposition"
    PERFORMANCE = "performance"
    HEALTH = "health"
    ALERTS = "alerts"
    TRADING = "trading"
    DIAGNOSTICS = "diagnostics"

class MetricType(Enum):
    """Metric types for dashboard."""
    GAUGE = "gauge"
    COUNTER = "counter"
    HISTOGRAM = "histogram"
    TIMESERIES = "timeseries"

@dataclass
class DashboardMetric:
    """Dashboard metric data structure."""
    name: str
    value: float
    unit: str
    metric_type: MetricType
    timestamp: datetime
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'name': self.name,
            'value': self.value,
            'unit': self.unit,
            'metric_type': self.metric_type.value,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata or {}
        }

class DashboardDataProvider:
    """Provides data for dashboard components."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.redis_client = redis.Redis(**config.get('redis', {}))
        self.metrics_cache = {}
        self.cache_ttl = config.get('cache_ttl', 30)  # seconds
        
        # Initialize monitoring components
        self.health_system = None
        self.superposition_monitor = None
        self.alert_manager = None
        self.diagnostic_suite = None
        
    def set_monitoring_components(self, health_system=None, superposition_monitor=None, 
                                alert_manager=None, diagnostic_suite=None):
        """Set monitoring component references."""
        self.health_system = health_system
        self.superposition_monitor = superposition_monitor
        self.alert_manager = alert_manager
        self.diagnostic_suite = diagnostic_suite
    
    async def get_system_overview(self) -> Dict[str, Any]:
        """Get system overview metrics."""
        try:
            # System health
            health_summary = {}
            if self.health_system:
                health_summary = self.health_system.get_health_summary()
            
            # Superposition status
            superposition_status = {}
            if self.superposition_monitor:
                superposition_status = self.superposition_monitor.get_monitoring_status()
            
            # Alert summary
            alert_stats = {}
            if self.alert_manager:
                alert_stats = self.alert_manager.get_alert_statistics()
            
            # System resources
            system_resources = self._get_system_resources()
            
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'health': health_summary,
                'superposition': superposition_status,
                'alerts': alert_stats,
                'resources': system_resources
            }
            
        except Exception as e:
            logger.error(f"Error getting system overview: {e}")
            return {'error': str(e)}
    
    async def get_superposition_metrics(self) -> Dict[str, Any]:
        """Get superposition-specific metrics."""
        try:
            if not self.superposition_monitor:
                return {'error': 'Superposition monitor not available'}
            
            # Get current superposition data
            monitoring_status = self.superposition_monitor.get_monitoring_status()
            performance_metrics = self.superposition_monitor.performance_metrics
            
            # Calculate aggregated metrics
            aggregated_metrics = self._calculate_superposition_aggregates(performance_metrics)
            
            # Get recent measurements
            recent_measurements = await self._get_recent_superposition_measurements()
            
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'monitoring_status': monitoring_status,
                'performance_metrics': performance_metrics,
                'aggregated_metrics': aggregated_metrics,
                'recent_measurements': recent_measurements
            }
            
        except Exception as e:
            logger.error(f"Error getting superposition metrics: {e}")
            return {'error': str(e)}
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        try:
            # Trading performance
            trading_metrics = await self._get_trading_performance()
            
            # System performance
            system_performance = self._get_system_performance()
            
            # MARL coordination metrics
            marl_metrics = await self._get_marl_coordination_metrics()
            
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'trading': trading_metrics,
                'system': system_performance,
                'marl': marl_metrics
            }
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {'error': str(e)}
    
    async def get_health_metrics(self) -> Dict[str, Any]:
        """Get health metrics."""
        try:
            if not self.health_system:
                return {'error': 'Health system not available'}
            
            # Get health summary
            health_summary = self.health_system.get_health_summary()
            
            # Get detailed health data
            health_details = {}
            for service_name in self.health_system.health_checkers.keys():
                if service_name in self.health_system.health_history:
                    history = self.health_system.health_history[service_name][-10:]  # Last 10 measurements
                    health_details[service_name] = [
                        {
                            'status': h.status.value,
                            'duration_ms': h.duration_ms,
                            'timestamp': h.timestamp.isoformat(),
                            'message': h.message
                        } for h in history
                    ]
            
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'summary': health_summary,
                'details': health_details
            }
            
        except Exception as e:
            logger.error(f"Error getting health metrics: {e}")
            return {'error': str(e)}
    
    async def get_alert_metrics(self) -> Dict[str, Any]:
        """Get alert metrics."""
        try:
            if not self.alert_manager:
                return {'error': 'Alert manager not available'}
            
            # Get alert statistics
            alert_stats = self.alert_manager.get_alert_statistics()
            
            # Get active alerts
            active_alerts = [alert.to_dict() for alert in self.alert_manager.get_active_alerts()]
            
            # Get recent alerts from history
            recent_alerts = []
            if hasattr(self.alert_manager, 'alert_history'):
                recent_alerts = [alert.to_dict() for alert in self.alert_manager.alert_history[-20:]]
            
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'statistics': alert_stats,
                'active_alerts': active_alerts,
                'recent_alerts': recent_alerts
            }
            
        except Exception as e:
            logger.error(f"Error getting alert metrics: {e}")
            return {'error': str(e)}
    
    def _get_system_resources(self) -> Dict[str, Any]:
        """Get system resource metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            
            # Memory usage
            memory = psutil.virtual_memory()
            
            # Disk usage
            disk = psutil.disk_usage('/')
            
            # Network stats
            network = psutil.net_io_counters()
            
            # Load average
            load_avg = psutil.getloadavg()
            
            return {
                'cpu': {
                    'percent': cpu_percent,
                    'count': cpu_count,
                    'load_avg': {
                        '1m': load_avg[0],
                        '5m': load_avg[1],
                        '15m': load_avg[2]
                    }
                },
                'memory': {
                    'percent': memory.percent,
                    'total_gb': memory.total / (1024**3),
                    'available_gb': memory.available / (1024**3),
                    'used_gb': memory.used / (1024**3)
                },
                'disk': {
                    'percent': (disk.used / disk.total) * 100,
                    'total_gb': disk.total / (1024**3),
                    'free_gb': disk.free / (1024**3),
                    'used_gb': disk.used / (1024**3)
                },
                'network': {
                    'bytes_sent': network.bytes_sent,
                    'bytes_recv': network.bytes_recv,
                    'packets_sent': network.packets_sent,
                    'packets_recv': network.packets_recv
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting system resources: {e}")
            return {}
    
    def _calculate_superposition_aggregates(self, performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate aggregated superposition metrics."""
        try:
            if not performance_metrics:
                return {}
            
            # Extract metrics by agent
            agent_metrics = {}
            for agent_id, metrics in performance_metrics.items():
                agent_metrics[agent_id] = {
                    'effectiveness': metrics.get('effectiveness', 0),
                    'coherence': metrics.get('coherence', 0),
                    'fidelity': metrics.get('fidelity', 0),
                    'entropy': metrics.get('entropy', 0),
                    'execution_time': metrics.get('execution_time', 0),
                    'error_rate': metrics.get('error_rate', 0)
                }
            
            # Calculate aggregates
            if agent_metrics:
                values = list(agent_metrics.values())
                
                aggregates = {
                    'effectiveness': {
                        'mean': np.mean([v['effectiveness'] for v in values]),
                        'min': np.min([v['effectiveness'] for v in values]),
                        'max': np.max([v['effectiveness'] for v in values]),
                        'std': np.std([v['effectiveness'] for v in values])
                    },
                    'coherence': {
                        'mean': np.mean([v['coherence'] for v in values]),
                        'min': np.min([v['coherence'] for v in values]),
                        'max': np.max([v['coherence'] for v in values]),
                        'std': np.std([v['coherence'] for v in values])
                    },
                    'fidelity': {
                        'mean': np.mean([v['fidelity'] for v in values]),
                        'min': np.min([v['fidelity'] for v in values]),
                        'max': np.max([v['fidelity'] for v in values]),
                        'std': np.std([v['fidelity'] for v in values])
                    },
                    'total_agents': len(agent_metrics),
                    'healthy_agents': sum(1 for v in values if v['effectiveness'] > 0.7)
                }
                
                return aggregates
            
            return {}
            
        except Exception as e:
            logger.error(f"Error calculating superposition aggregates: {e}")
            return {}
    
    async def _get_recent_superposition_measurements(self) -> List[Dict[str, Any]]:
        """Get recent superposition measurements."""
        try:
            # This would typically query the measurement storage
            # For now, return placeholder data
            measurements = []
            
            # Get from Redis if available
            for i in range(10):  # Last 10 measurements
                measurement_key = f"superposition_measurement:{i}"
                measurement_data = await self.redis_client.get(measurement_key)
                if measurement_data:
                    measurements.append(json.loads(measurement_data))
            
            return measurements
            
        except Exception as e:
            logger.error(f"Error getting recent measurements: {e}")
            return []
    
    async def _get_trading_performance(self) -> Dict[str, Any]:
        """Get trading performance metrics."""
        try:
            # This would typically query the trading system
            # For now, return placeholder data
            return {
                'pnl': {
                    'daily': 5000,
                    'weekly': 25000,
                    'monthly': 100000
                },
                'trades': {
                    'total': 1250,
                    'successful': 1125,
                    'success_rate': 0.90
                },
                'risk': {
                    'var_95': 0.025,
                    'max_drawdown': 0.15,
                    'sharpe_ratio': 2.1
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting trading performance: {e}")
            return {}
    
    def _get_system_performance(self) -> Dict[str, Any]:
        """Get system performance metrics."""
        try:
            # Latency metrics
            latency_metrics = {
                'inference_latency_ms': 15.5,
                'execution_latency_ms': 8.2,
                'total_latency_ms': 23.7
            }
            
            # Throughput metrics
            throughput_metrics = {
                'inferences_per_second': 120,
                'trades_per_minute': 5,
                'data_processing_rate': 1000
            }
            
            # Error rates
            error_rates = {
                'inference_error_rate': 0.001,
                'execution_error_rate': 0.0005,
                'data_error_rate': 0.0001
            }
            
            return {
                'latency': latency_metrics,
                'throughput': throughput_metrics,
                'error_rates': error_rates
            }
            
        except Exception as e:
            logger.error(f"Error getting system performance: {e}")
            return {}
    
    async def _get_marl_coordination_metrics(self) -> Dict[str, Any]:
        """Get MARL coordination metrics."""
        try:
            # This would typically query the MARL system
            # For now, return placeholder data
            return {
                'coordination_score': 0.85,
                'consensus_rate': 0.92,
                'disagreement_rate': 0.08,
                'convergence_time_ms': 250,
                'agent_synchronization': 0.88
            }
            
        except Exception as e:
            logger.error(f"Error getting MARL coordination metrics: {e}")
            return {}

class DashboardVisualizationGenerator:
    """Generates visualizations for dashboard."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.color_scheme = config.get('color_scheme', 'plotly')
        
    def create_overview_dashboard(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create overview dashboard visualization."""
        try:
            # System health gauge
            health_gauge = self._create_health_gauge(data.get('health', {}))
            
            # Resource usage charts
            resource_charts = self._create_resource_charts(data.get('resources', {}))
            
            # Alert timeline
            alert_timeline = self._create_alert_timeline(data.get('alerts', {}))
            
            # Superposition effectiveness
            superposition_chart = self._create_superposition_overview(data.get('superposition', {}))
            
            return {
                'health_gauge': health_gauge,
                'resource_charts': resource_charts,
                'alert_timeline': alert_timeline,
                'superposition_chart': superposition_chart
            }
            
        except Exception as e:
            logger.error(f"Error creating overview dashboard: {e}")
            return {'error': str(e)}
    
    def create_superposition_dashboard(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create superposition dashboard visualization."""
        try:
            # Coherence time series
            coherence_chart = self._create_coherence_timeseries(data)
            
            # Fidelity vs Entropy scatter
            fidelity_entropy_scatter = self._create_fidelity_entropy_scatter(data)
            
            # Agent effectiveness comparison
            agent_comparison = self._create_agent_effectiveness_comparison(data)
            
            # Quantum state heatmap
            quantum_heatmap = self._create_quantum_state_heatmap(data)
            
            return {
                'coherence_chart': coherence_chart,
                'fidelity_entropy_scatter': fidelity_entropy_scatter,
                'agent_comparison': agent_comparison,
                'quantum_heatmap': quantum_heatmap
            }
            
        except Exception as e:
            logger.error(f"Error creating superposition dashboard: {e}")
            return {'error': str(e)}
    
    def create_performance_dashboard(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create performance dashboard visualization."""
        try:
            # Trading performance metrics
            trading_chart = self._create_trading_performance_chart(data.get('trading', {}))
            
            # System latency chart
            latency_chart = self._create_latency_chart(data.get('system', {}))
            
            # MARL coordination chart
            coordination_chart = self._create_coordination_chart(data.get('marl', {}))
            
            # Throughput vs Error rate
            throughput_error_chart = self._create_throughput_error_chart(data.get('system', {}))
            
            return {
                'trading_chart': trading_chart,
                'latency_chart': latency_chart,
                'coordination_chart': coordination_chart,
                'throughput_error_chart': throughput_error_chart
            }
            
        except Exception as e:
            logger.error(f"Error creating performance dashboard: {e}")
            return {'error': str(e)}
    
    def _create_health_gauge(self, health_data: Dict[str, Any]) -> str:
        """Create health gauge chart."""
        try:
            overall_status = health_data.get('overall_status', 'unknown')
            
            # Map status to numeric value
            status_map = {
                'healthy': 100,
                'degraded': 60,
                'unhealthy': 20,
                'unknown': 0
            }
            
            value = status_map.get(overall_status, 0)
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=value,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "System Health"},
                delta={'reference': 90},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 40], 'color': "red"},
                        {'range': [40, 70], 'color': "orange"},
                        {'range': [70, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            
        except Exception as e:
            logger.error(f"Error creating health gauge: {e}")
            return "{}"
    
    def _create_resource_charts(self, resource_data: Dict[str, Any]) -> str:
        """Create resource usage charts."""
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('CPU Usage', 'Memory Usage', 'Disk Usage', 'Network I/O'),
                specs=[[{'type': 'indicator'}, {'type': 'indicator'}],
                       [{'type': 'indicator'}, {'type': 'bar'}]]
            )
            
            # CPU gauge
            cpu_percent = resource_data.get('cpu', {}).get('percent', 0)
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=cpu_percent,
                title={'text': "CPU %"},
                gauge={'axis': {'range': [0, 100]},
                       'bar': {'color': "blue"},
                       'steps': [{'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 80], 'color': "yellow"},
                                {'range': [80, 100], 'color': "red"}]}
            ), row=1, col=1)
            
            # Memory gauge
            memory_percent = resource_data.get('memory', {}).get('percent', 0)
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=memory_percent,
                title={'text': "Memory %"},
                gauge={'axis': {'range': [0, 100]},
                       'bar': {'color': "green"},
                       'steps': [{'range': [0, 60], 'color': "lightgray"},
                                {'range': [60, 85], 'color': "yellow"},
                                {'range': [85, 100], 'color': "red"}]}
            ), row=1, col=2)
            
            # Disk gauge
            disk_percent = resource_data.get('disk', {}).get('percent', 0)
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=disk_percent,
                title={'text': "Disk %"},
                gauge={'axis': {'range': [0, 100]},
                       'bar': {'color': "orange"},
                       'steps': [{'range': [0, 70], 'color': "lightgray"},
                                {'range': [70, 90], 'color': "yellow"},
                                {'range': [90, 100], 'color': "red"}]}
            ), row=2, col=1)
            
            # Network bar chart
            network_data = resource_data.get('network', {})
            fig.add_trace(go.Bar(
                x=['Bytes Sent', 'Bytes Recv'],
                y=[network_data.get('bytes_sent', 0), network_data.get('bytes_recv', 0)],
                name='Network I/O'
            ), row=2, col=2)
            
            fig.update_layout(height=600, showlegend=False)
            
            return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            
        except Exception as e:
            logger.error(f"Error creating resource charts: {e}")
            return "{}"
    
    def _create_alert_timeline(self, alert_data: Dict[str, Any]) -> str:
        """Create alert timeline chart."""
        try:
            recent_alerts = alert_data.get('recent_alerts', [])
            
            if not recent_alerts:
                # Create empty chart
                fig = go.Figure()
                fig.add_annotation(
                    text="No recent alerts",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=16)
                )
                fig.update_layout(
                    title="Alert Timeline",
                    xaxis_title="Time",
                    yaxis_title="Alert Count"
                )
                return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            
            # Process alerts for timeline
            timestamps = [alert['timestamp'] for alert in recent_alerts]
            severities = [alert['severity'] for alert in recent_alerts]
            
            # Create timeline
            fig = go.Figure()
            
            severity_colors = {
                'low': 'green',
                'medium': 'yellow',
                'high': 'orange',
                'critical': 'red'
            }
            
            for i, (timestamp, severity) in enumerate(zip(timestamps, severities)):
                fig.add_trace(go.Scatter(
                    x=[timestamp],
                    y=[i],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=severity_colors.get(severity, 'gray')
                    ),
                    name=f"{severity} alert",
                    showlegend=False
                ))
            
            fig.update_layout(
                title="Alert Timeline",
                xaxis_title="Time",
                yaxis_title="Alert Index",
                height=300
            )
            
            return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            
        except Exception as e:
            logger.error(f"Error creating alert timeline: {e}")
            return "{}"
    
    def _create_superposition_overview(self, superposition_data: Dict[str, Any]) -> str:
        """Create superposition overview chart."""
        try:
            performance_metrics = superposition_data.get('performance_metrics', {})
            
            if not performance_metrics:
                # Create empty chart
                fig = go.Figure()
                fig.add_annotation(
                    text="No superposition data available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=16)
                )
                return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            
            # Create bar chart of agent effectiveness
            agents = list(performance_metrics.keys())
            effectiveness = [metrics.get('effectiveness', 0) for metrics in performance_metrics.values()]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=agents,
                    y=effectiveness,
                    marker_color=['red' if e < 0.7 else 'yellow' if e < 0.85 else 'green' for e in effectiveness]
                )
            ])
            
            fig.update_layout(
                title="Agent Superposition Effectiveness",
                xaxis_title="Agent",
                yaxis_title="Effectiveness",
                yaxis_range=[0, 1]
            )
            
            return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            
        except Exception as e:
            logger.error(f"Error creating superposition overview: {e}")
            return "{}"
    
    def _create_coherence_timeseries(self, data: Dict[str, Any]) -> str:
        """Create coherence time series chart."""
        try:
            performance_metrics = data.get('performance_metrics', {})
            
            fig = go.Figure()
            
            for agent_id, metrics in performance_metrics.items():
                coherence = metrics.get('coherence', 0)
                timestamp = metrics.get('timestamp', datetime.utcnow().isoformat())
                
                fig.add_trace(go.Scatter(
                    x=[timestamp],
                    y=[coherence],
                    mode='lines+markers',
                    name=f"{agent_id} Coherence",
                    line=dict(width=2)
                ))
            
            fig.update_layout(
                title="Quantum Coherence Time Series",
                xaxis_title="Time",
                yaxis_title="Coherence",
                yaxis_range=[0, 1]
            )
            
            return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            
        except Exception as e:
            logger.error(f"Error creating coherence timeseries: {e}")
            return "{}"
    
    def _create_fidelity_entropy_scatter(self, data: Dict[str, Any]) -> str:
        """Create fidelity vs entropy scatter plot."""
        try:
            performance_metrics = data.get('performance_metrics', {})
            
            agents = []
            fidelities = []
            entropies = []
            
            for agent_id, metrics in performance_metrics.items():
                agents.append(agent_id)
                fidelities.append(metrics.get('fidelity', 0))
                entropies.append(metrics.get('entropy', 0))
            
            fig = go.Figure(data=go.Scatter(
                x=fidelities,
                y=entropies,
                mode='markers',
                marker=dict(
                    size=12,
                    color=fidelities,
                    colorscale='viridis',
                    colorbar=dict(title="Fidelity")
                ),
                text=agents,
                textposition="top center"
            ))
            
            fig.update_layout(
                title="Fidelity vs Entropy",
                xaxis_title="Fidelity",
                yaxis_title="Entropy"
            )
            
            return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            
        except Exception as e:
            logger.error(f"Error creating fidelity entropy scatter: {e}")
            return "{}"
    
    def _create_agent_effectiveness_comparison(self, data: Dict[str, Any]) -> str:
        """Create agent effectiveness comparison."""
        try:
            performance_metrics = data.get('performance_metrics', {})
            
            agents = list(performance_metrics.keys())
            effectiveness = [metrics.get('effectiveness', 0) for metrics in performance_metrics.values()]
            coherence = [metrics.get('coherence', 0) for metrics in performance_metrics.values()]
            fidelity = [metrics.get('fidelity', 0) for metrics in performance_metrics.values()]
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=agents,
                y=effectiveness,
                name='Effectiveness',
                marker_color='blue'
            ))
            
            fig.add_trace(go.Bar(
                x=agents,
                y=coherence,
                name='Coherence',
                marker_color='red'
            ))
            
            fig.add_trace(go.Bar(
                x=agents,
                y=fidelity,
                name='Fidelity',
                marker_color='green'
            ))
            
            fig.update_layout(
                title="Agent Performance Comparison",
                xaxis_title="Agent",
                yaxis_title="Score",
                barmode='group'
            )
            
            return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            
        except Exception as e:
            logger.error(f"Error creating agent effectiveness comparison: {e}")
            return "{}"
    
    def _create_quantum_state_heatmap(self, data: Dict[str, Any]) -> str:
        """Create quantum state heatmap."""
        try:
            performance_metrics = data.get('performance_metrics', {})
            
            if not performance_metrics:
                # Create empty heatmap
                fig = go.Figure()
                fig.add_annotation(
                    text="No quantum state data available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=16)
                )
                return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            
            # Create heatmap matrix
            agents = list(performance_metrics.keys())
            metrics = ['effectiveness', 'coherence', 'fidelity', 'entropy']
            
            z_data = []
            for metric in metrics:
                row = []
                for agent_id in agents:
                    value = performance_metrics[agent_id].get(metric, 0)
                    row.append(value)
                z_data.append(row)
            
            fig = go.Figure(data=go.Heatmap(
                z=z_data,
                x=agents,
                y=metrics,
                colorscale='RdYlBu',
                colorbar=dict(title="Value")
            ))
            
            fig.update_layout(
                title="Quantum State Heatmap",
                xaxis_title="Agent",
                yaxis_title="Metric"
            )
            
            return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            
        except Exception as e:
            logger.error(f"Error creating quantum state heatmap: {e}")
            return "{}"
    
    def _create_trading_performance_chart(self, trading_data: Dict[str, Any]) -> str:
        """Create trading performance chart."""
        try:
            pnl_data = trading_data.get('pnl', {})
            
            periods = ['Daily', 'Weekly', 'Monthly']
            values = [pnl_data.get('daily', 0), pnl_data.get('weekly', 0), pnl_data.get('monthly', 0)]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=periods,
                    y=values,
                    marker_color=['green' if v > 0 else 'red' for v in values]
                )
            ])
            
            fig.update_layout(
                title="Trading Performance (PnL)",
                xaxis_title="Period",
                yaxis_title="PnL ($)"
            )
            
            return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            
        except Exception as e:
            logger.error(f"Error creating trading performance chart: {e}")
            return "{}"
    
    def _create_latency_chart(self, system_data: Dict[str, Any]) -> str:
        """Create latency chart."""
        try:
            latency_data = system_data.get('latency', {})
            
            components = ['Inference', 'Execution', 'Total']
            latencies = [
                latency_data.get('inference_latency_ms', 0),
                latency_data.get('execution_latency_ms', 0),
                latency_data.get('total_latency_ms', 0)
            ]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=components,
                    y=latencies,
                    marker_color=['blue', 'orange', 'red']
                )
            ])
            
            fig.update_layout(
                title="System Latency",
                xaxis_title="Component",
                yaxis_title="Latency (ms)"
            )
            
            return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            
        except Exception as e:
            logger.error(f"Error creating latency chart: {e}")
            return "{}"
    
    def _create_coordination_chart(self, marl_data: Dict[str, Any]) -> str:
        """Create MARL coordination chart."""
        try:
            metrics = ['Coordination Score', 'Consensus Rate', 'Synchronization']
            values = [
                marl_data.get('coordination_score', 0),
                marl_data.get('consensus_rate', 0),
                marl_data.get('agent_synchronization', 0)
            ]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=metrics,
                    y=values,
                    marker_color=['green', 'blue', 'orange']
                )
            ])
            
            fig.update_layout(
                title="MARL Coordination Metrics",
                xaxis_title="Metric",
                yaxis_title="Score",
                yaxis_range=[0, 1]
            )
            
            return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            
        except Exception as e:
            logger.error(f"Error creating coordination chart: {e}")
            return "{}"
    
    def _create_throughput_error_chart(self, system_data: Dict[str, Any]) -> str:
        """Create throughput vs error rate chart."""
        try:
            throughput_data = system_data.get('throughput', {})
            error_data = system_data.get('error_rates', {})
            
            # Create subplot with secondary y-axis
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Throughput bars
            fig.add_trace(
                go.Bar(
                    x=['Inferences/sec', 'Trades/min', 'Data Processing'],
                    y=[throughput_data.get('inferences_per_second', 0),
                       throughput_data.get('trades_per_minute', 0),
                       throughput_data.get('data_processing_rate', 0)],
                    name="Throughput",
                    marker_color='blue'
                ),
                secondary_y=False
            )
            
            # Error rate line
            fig.add_trace(
                go.Scatter(
                    x=['Inference', 'Execution', 'Data'],
                    y=[error_data.get('inference_error_rate', 0),
                       error_data.get('execution_error_rate', 0),
                       error_data.get('data_error_rate', 0)],
                    name="Error Rate",
                    line=dict(color='red', width=3),
                    mode='lines+markers'
                ),
                secondary_y=True
            )
            
            fig.update_xaxes(title_text="Component")
            fig.update_yaxes(title_text="Throughput", secondary_y=False)
            fig.update_yaxes(title_text="Error Rate", secondary_y=True)
            
            fig.update_layout(title="Throughput vs Error Rate")
            
            return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            
        except Exception as e:
            logger.error(f"Error creating throughput error chart: {e}")
            return "{}"

class DashboardServer:
    """Dashboard web server."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.app = FastAPI(title="GrandModel Monitoring Dashboard")
        self.data_provider = DashboardDataProvider(config.get('data_provider', {}))
        self.viz_generator = DashboardVisualizationGenerator(config.get('visualization', {}))
        self.websocket_connections = set()
        
        # Setup routes
        self._setup_routes()
        
        # Start background tasks
        self.background_tasks = []
        
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/")
        async def root():
            return {"message": "GrandModel Monitoring Dashboard"}
        
        @self.app.get("/health")
        async def health():
            return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
        
        @self.app.get("/api/overview")
        async def get_overview():
            DASHBOARD_VIEWS.labels(dashboard_type=DashboardType.OVERVIEW.value).inc()
            data = await self.data_provider.get_system_overview()
            return JSONResponse(content=data)
        
        @self.app.get("/api/superposition")
        async def get_superposition():
            DASHBOARD_VIEWS.labels(dashboard_type=DashboardType.SUPERPOSITION.value).inc()
            data = await self.data_provider.get_superposition_metrics()
            return JSONResponse(content=data)
        
        @self.app.get("/api/performance")
        async def get_performance():
            DASHBOARD_VIEWS.labels(dashboard_type=DashboardType.PERFORMANCE.value).inc()
            data = await self.data_provider.get_performance_metrics()
            return JSONResponse(content=data)
        
        @self.app.get("/api/health-metrics")
        async def get_health_metrics():
            DASHBOARD_VIEWS.labels(dashboard_type=DashboardType.HEALTH.value).inc()
            data = await self.data_provider.get_health_metrics()
            return JSONResponse(content=data)
        
        @self.app.get("/api/alerts")
        async def get_alerts():
            DASHBOARD_VIEWS.labels(dashboard_type=DashboardType.ALERTS.value).inc()
            data = await self.data_provider.get_alert_metrics()
            return JSONResponse(content=data)
        
        @self.app.get("/api/visualizations/overview")
        async def get_overview_visualizations():
            data = await self.data_provider.get_system_overview()
            visualizations = self.viz_generator.create_overview_dashboard(data)
            return JSONResponse(content=visualizations)
        
        @self.app.get("/api/visualizations/superposition")
        async def get_superposition_visualizations():
            data = await self.data_provider.get_superposition_metrics()
            visualizations = self.viz_generator.create_superposition_dashboard(data)
            return JSONResponse(content=visualizations)
        
        @self.app.get("/api/visualizations/performance")
        async def get_performance_visualizations():
            data = await self.data_provider.get_performance_metrics()
            visualizations = self.viz_generator.create_performance_dashboard(data)
            return JSONResponse(content=visualizations)
        
        @self.app.get("/metrics")
        async def get_prometheus_metrics():
            return generate_latest().decode('utf-8')
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self.websocket_connections.add(websocket)
            ACTIVE_WEBSOCKETS.inc()
            
            try:
                while True:
                    # Send real-time updates
                    overview_data = await self.data_provider.get_system_overview()
                    await websocket.send_json({
                        'type': 'overview_update',
                        'data': overview_data
                    })
                    
                    await asyncio.sleep(5)  # Update every 5 seconds
                    
            except WebSocketDisconnect:
                pass
            finally:
                self.websocket_connections.discard(websocket)
                ACTIVE_WEBSOCKETS.dec()
    
    async def start_server(self, host: str = "0.0.0.0", port: int = 8080):
        """Start the dashboard server."""
        logger.info(f"Starting dashboard server on {host}:{port}")
        
        # Start background tasks
        await self._start_background_tasks()
        
        # Start server
        config = uvicorn.Config(
            self.app,
            host=host,
            port=port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()
    
    async def _start_background_tasks(self):
        """Start background tasks."""
        # WebSocket broadcast task
        task = asyncio.create_task(self._websocket_broadcast_task())
        self.background_tasks.append(task)
    
    async def _websocket_broadcast_task(self):
        """Broadcast updates to WebSocket connections."""
        while True:
            try:
                if self.websocket_connections:
                    # Get latest data
                    overview_data = await self.data_provider.get_system_overview()
                    
                    # Broadcast to all connections
                    disconnected = set()
                    for websocket in self.websocket_connections:
                        try:
                            await websocket.send_json({
                                'type': 'overview_update',
                                'data': overview_data,
                                'timestamp': datetime.utcnow().isoformat()
                            })
                        except:
                            disconnected.add(websocket)
                    
                    # Remove disconnected websockets
                    for websocket in disconnected:
                        self.websocket_connections.discard(websocket)
                        ACTIVE_WEBSOCKETS.dec()
                
                await asyncio.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in websocket broadcast task: {e}")
                await asyncio.sleep(5)

# Factory function
def create_dashboard_server(config: Dict[str, Any]) -> DashboardServer:
    """Create dashboard server instance."""
    return DashboardServer(config)

# Example configuration
EXAMPLE_CONFIG = {
    'data_provider': {
        'redis': {
            'host': 'localhost',
            'port': 6379,
            'db': 0
        },
        'cache_ttl': 30
    },
    'visualization': {
        'color_scheme': 'plotly'
    },
    'server': {
        'host': '0.0.0.0',
        'port': 8080
    }
}

# Example usage
async def main():
    """Example usage of dashboard server."""
    config = EXAMPLE_CONFIG
    dashboard = create_dashboard_server(config)
    
    # Start server
    await dashboard.start_server(
        host=config['server']['host'],
        port=config['server']['port']
    )

if __name__ == "__main__":
    asyncio.run(main())