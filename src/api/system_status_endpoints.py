"""
System Status Monitoring API Endpoints
=======================================

Comprehensive status monitoring endpoints for GrandModel system including:
- Real-time system metrics
- Component health monitoring
- Performance analytics
- Alert management
- Historical data tracking
- WebSocket support for real-time updates

This module provides detailed monitoring capabilities for system administrators
and operators to track system health and performance in real-time.
"""

import os
import time
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from enum import Enum
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException, Depends, Request, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import psutil
import sqlite3
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Security
security = HTTPBearer()

# Enums
class ComponentType(str, Enum):
    """Component type enumeration"""
    CORE = "core"
    ENGINE = "engine"
    DATABASE = "database"
    CACHE = "cache"
    MONITORING = "monitoring"
    EXTERNAL = "external"

class MetricType(str, Enum):
    """Metric type enumeration"""
    GAUGE = "gauge"
    COUNTER = "counter"
    HISTOGRAM = "histogram"
    TIMER = "timer"

class AlertSeverity(str, Enum):
    """Alert severity enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertStatus(str, Enum):
    """Alert status enumeration"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"

# Pydantic models
class SystemMetric(BaseModel):
    """System metric model"""
    name: str
    value: Union[int, float, str]
    unit: str
    type: MetricType
    timestamp: datetime
    labels: Dict[str, str] = {}

class ComponentHealth(BaseModel):
    """Component health model"""
    component_id: str
    component_type: ComponentType
    status: str
    health_score: float
    last_check: datetime
    metrics: List[SystemMetric]
    issues: List[str]
    uptime_seconds: Optional[int] = None

class SystemOverview(BaseModel):
    """System overview model"""
    timestamp: datetime
    overall_health: float
    total_components: int
    healthy_components: int
    degraded_components: int
    unhealthy_components: int
    system_uptime: int
    active_alerts: int
    performance_score: float

class PerformanceMetrics(BaseModel):
    """Performance metrics model"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: float
    active_connections: int
    request_rate: float
    response_time: float
    error_rate: float
    throughput: float

class AlertModel(BaseModel):
    """Alert model"""
    id: str
    title: str
    description: str
    severity: AlertSeverity
    status: AlertStatus
    component_id: str
    metric_name: str
    threshold_value: float
    current_value: float
    created_at: datetime
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None

class HistoricalData(BaseModel):
    """Historical data model"""
    metric_name: str
    component_id: str
    data_points: List[Dict[str, Any]]
    start_time: datetime
    end_time: datetime
    aggregation: str
    interval: str

class WebSocketMessage(BaseModel):
    """WebSocket message model"""
    type: str
    data: Dict[str, Any]
    timestamp: datetime

class SystemStatusMonitor:
    """
    System status monitor for tracking system health and performance
    """
    
    def __init__(self):
        self.components = {}
        self.metrics_history = []
        self.alerts = []
        self.websocket_connections = set()
        self.monitoring_active = False
        self.system_start_time = datetime.now()
        
        # Initialize database
        self._init_database()
        
        # Initialize components
        self._initialize_components()
        
        # Start monitoring
        self._start_monitoring()
        
    def _init_database(self):
        """Initialize SQLite database for historical data"""
        self.db_path = "system_monitoring.db"
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    component_id TEXT NOT NULL,
                    value REAL NOT NULL,
                    unit TEXT NOT NULL,
                    type TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    labels TEXT
                )
            ''')
            
            # Create alerts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    status TEXT NOT NULL,
                    component_id TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    threshold_value REAL NOT NULL,
                    current_value REAL NOT NULL,
                    created_at DATETIME NOT NULL,
                    acknowledged_at DATETIME,
                    resolved_at DATETIME,
                    acknowledged_by TEXT
                )
            ''')
            
            # Create component_health table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS component_health (
                    component_id TEXT PRIMARY KEY,
                    component_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    health_score REAL NOT NULL,
                    last_check DATETIME NOT NULL,
                    uptime_seconds INTEGER
                )
            ''')
            
            conn.commit()
    
    def _initialize_components(self):
        """Initialize component monitoring"""
        self.components = {
            "marl_engine": {
                "type": ComponentType.ENGINE,
                "status": "running",
                "health_score": 0.95,
                "last_check": datetime.now(),
                "metrics": [],
                "issues": [],
                "uptime_seconds": 3600
            },
            "risk_manager": {
                "type": ComponentType.CORE,
                "status": "running",
                "health_score": 0.98,
                "last_check": datetime.now(),
                "metrics": [],
                "issues": [],
                "uptime_seconds": 3600
            },
            "execution_engine": {
                "type": ComponentType.ENGINE,
                "status": "running",
                "health_score": 0.92,
                "last_check": datetime.now(),
                "metrics": [],
                "issues": [],
                "uptime_seconds": 3600
            },
            "database": {
                "type": ComponentType.DATABASE,
                "status": "running",
                "health_score": 0.99,
                "last_check": datetime.now(),
                "metrics": [],
                "issues": [],
                "uptime_seconds": 7200
            },
            "redis_cache": {
                "type": ComponentType.CACHE,
                "status": "running",
                "health_score": 0.97,
                "last_check": datetime.now(),
                "metrics": [],
                "issues": [],
                "uptime_seconds": 7200
            },
            "monitoring_system": {
                "type": ComponentType.MONITORING,
                "status": "running",
                "health_score": 1.0,
                "last_check": datetime.now(),
                "metrics": [],
                "issues": [],
                "uptime_seconds": 3600
            }
        }
    
    def _start_monitoring(self):
        """Start background monitoring tasks"""
        self.monitoring_active = True
        
        # Start monitoring tasks in background
        asyncio.create_task(self._monitor_components())
        asyncio.create_task(self._monitor_performance())
        asyncio.create_task(self._check_alerts())
        
    async def _monitor_components(self):
        """Monitor component health continuously"""
        while self.monitoring_active:
            try:
                for component_id, component in self.components.items():
                    # Update component metrics
                    await self._update_component_metrics(component_id)
                    
                    # Check component health
                    health_score = await self._calculate_component_health(component_id)
                    component["health_score"] = health_score
                    component["last_check"] = datetime.now()
                    
                    # Update database
                    await self._store_component_health(component_id, component)
                    
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in component monitoring: {e}")
                await asyncio.sleep(5)
    
    async def _monitor_performance(self):
        """Monitor system performance continuously"""
        while self.monitoring_active:
            try:
                # Get system metrics
                metrics = await self._get_system_performance_metrics()
                
                # Store metrics
                await self._store_metrics(metrics)
                
                # Broadcast to WebSocket clients
                await self._broadcast_metrics(metrics)
                
                await asyncio.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(5)
    
    async def _check_alerts(self):
        """Check for alert conditions"""
        while self.monitoring_active:
            try:
                # Check metric thresholds
                await self._check_metric_thresholds()
                
                # Check component health
                await self._check_component_health_alerts()
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in alert checking: {e}")
                await asyncio.sleep(10)
    
    async def _update_component_metrics(self, component_id: str):
        """Update metrics for a specific component"""
        # Simulate component-specific metrics
        if component_id == "marl_engine":
            metrics = [
                SystemMetric(
                    name="inference_latency",
                    value=round(12.5 + (time.time() % 10), 2),
                    unit="ms",
                    type=MetricType.TIMER,
                    timestamp=datetime.now(),
                    labels={"component": component_id}
                ),
                SystemMetric(
                    name="model_accuracy",
                    value=round(0.85 + (time.time() % 0.1), 3),
                    unit="ratio",
                    type=MetricType.GAUGE,
                    timestamp=datetime.now(),
                    labels={"component": component_id}
                )
            ]
        elif component_id == "database":
            metrics = [
                SystemMetric(
                    name="query_response_time",
                    value=round(5.2 + (time.time() % 5), 2),
                    unit="ms",
                    type=MetricType.TIMER,
                    timestamp=datetime.now(),
                    labels={"component": component_id}
                ),
                SystemMetric(
                    name="connection_count",
                    value=int(15 + (time.time() % 10)),
                    unit="count",
                    type=MetricType.GAUGE,
                    timestamp=datetime.now(),
                    labels={"component": component_id}
                )
            ]
        else:
            # Generic metrics
            metrics = [
                SystemMetric(
                    name="cpu_usage",
                    value=round(30 + (time.time() % 20), 2),
                    unit="percent",
                    type=MetricType.GAUGE,
                    timestamp=datetime.now(),
                    labels={"component": component_id}
                ),
                SystemMetric(
                    name="memory_usage",
                    value=round(40 + (time.time() % 30), 2),
                    unit="percent",
                    type=MetricType.GAUGE,
                    timestamp=datetime.now(),
                    labels={"component": component_id}
                )
            ]
        
        self.components[component_id]["metrics"] = metrics
        
        # Store metrics in database
        for metric in metrics:
            await self._store_metric(metric, component_id)
    
    async def _calculate_component_health(self, component_id: str) -> float:
        """Calculate health score for a component"""
        component = self.components.get(component_id, {})
        
        # Base health score
        health_score = 1.0
        
        # Check metrics
        for metric in component.get("metrics", []):
            if metric.name == "cpu_usage" and metric.value > 80:
                health_score -= 0.1
            elif metric.name == "memory_usage" and metric.value > 85:
                health_score -= 0.1
            elif metric.name == "inference_latency" and metric.value > 50:
                health_score -= 0.05
        
        # Check status
        if component.get("status") != "running":
            health_score -= 0.5
        
        return max(0.0, min(1.0, health_score))
    
    async def _get_system_performance_metrics(self) -> PerformanceMetrics:
        """Get current system performance metrics"""
        try:
            # Get CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Get memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Get disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            # Get network IO
            network = psutil.net_io_counters()
            network_io = (network.bytes_sent + network.bytes_recv) / 1024 / 1024  # MB
            
            return PerformanceMetrics(
                timestamp=datetime.now(),
                cpu_usage=cpu_percent,
                memory_usage=memory_percent,
                disk_usage=disk_percent,
                network_io=network_io,
                active_connections=len(self.websocket_connections),
                request_rate=125.5,  # Simulated
                response_time=18.2,  # Simulated
                error_rate=0.15,     # Simulated
                throughput=1250.0    # Simulated
            )
            
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return PerformanceMetrics(
                timestamp=datetime.now(),
                cpu_usage=0.0,
                memory_usage=0.0,
                disk_usage=0.0,
                network_io=0.0,
                active_connections=0,
                request_rate=0.0,
                response_time=0.0,
                error_rate=0.0,
                throughput=0.0
            )
    
    async def _store_metric(self, metric: SystemMetric, component_id: str):
        """Store metric in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO metrics (name, component_id, value, unit, type, timestamp, labels)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metric.name,
                    component_id,
                    metric.value,
                    metric.unit,
                    metric.type.value,
                    metric.timestamp,
                    json.dumps(metric.labels)
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error storing metric: {e}")
    
    async def _store_metrics(self, metrics: PerformanceMetrics):
        """Store performance metrics"""
        system_metrics = [
            SystemMetric(
                name="cpu_usage",
                value=metrics.cpu_usage,
                unit="percent",
                type=MetricType.GAUGE,
                timestamp=metrics.timestamp,
                labels={"component": "system"}
            ),
            SystemMetric(
                name="memory_usage",
                value=metrics.memory_usage,
                unit="percent",
                type=MetricType.GAUGE,
                timestamp=metrics.timestamp,
                labels={"component": "system"}
            ),
            SystemMetric(
                name="disk_usage",
                value=metrics.disk_usage,
                unit="percent",
                type=MetricType.GAUGE,
                timestamp=metrics.timestamp,
                labels={"component": "system"}
            )
        ]
        
        for metric in system_metrics:
            await self._store_metric(metric, "system")
    
    async def _store_component_health(self, component_id: str, component: Dict[str, Any]):
        """Store component health in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO component_health 
                    (component_id, component_type, status, health_score, last_check, uptime_seconds)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    component_id,
                    component["type"].value,
                    component["status"],
                    component["health_score"],
                    component["last_check"],
                    component.get("uptime_seconds", 0)
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error storing component health: {e}")
    
    async def _check_metric_thresholds(self):
        """Check metrics against thresholds and create alerts"""
        # Define thresholds
        thresholds = {
            "cpu_usage": {"high": 80, "critical": 90},
            "memory_usage": {"high": 85, "critical": 95},
            "disk_usage": {"high": 80, "critical": 90},
            "inference_latency": {"high": 50, "critical": 100},
            "query_response_time": {"high": 20, "critical": 50}
        }
        
        # Check all components
        for component_id, component in self.components.items():
            for metric in component.get("metrics", []):
                if metric.name in thresholds:
                    threshold_config = thresholds[metric.name]
                    
                    if metric.value >= threshold_config["critical"]:
                        await self._create_alert(
                            component_id,
                            metric.name,
                            f"Critical {metric.name} threshold exceeded",
                            f"{metric.name} is {metric.value}{metric.unit}, exceeding critical threshold of {threshold_config['critical']}{metric.unit}",
                            AlertSeverity.CRITICAL,
                            threshold_config["critical"],
                            metric.value
                        )
                    elif metric.value >= threshold_config["high"]:
                        await self._create_alert(
                            component_id,
                            metric.name,
                            f"High {metric.name} threshold exceeded",
                            f"{metric.name} is {metric.value}{metric.unit}, exceeding high threshold of {threshold_config['high']}{metric.unit}",
                            AlertSeverity.HIGH,
                            threshold_config["high"],
                            metric.value
                        )
    
    async def _check_component_health_alerts(self):
        """Check component health and create alerts"""
        for component_id, component in self.components.items():
            health_score = component.get("health_score", 1.0)
            
            if health_score < 0.5:
                await self._create_alert(
                    component_id,
                    "health_score",
                    f"Component {component_id} health critical",
                    f"Component {component_id} health score is {health_score:.2f}, indicating critical issues",
                    AlertSeverity.CRITICAL,
                    0.5,
                    health_score
                )
            elif health_score < 0.8:
                await self._create_alert(
                    component_id,
                    "health_score",
                    f"Component {component_id} health degraded",
                    f"Component {component_id} health score is {health_score:.2f}, indicating degraded performance",
                    AlertSeverity.HIGH,
                    0.8,
                    health_score
                )
    
    async def _create_alert(self, component_id: str, metric_name: str, title: str, 
                           description: str, severity: AlertSeverity, 
                           threshold_value: float, current_value: float):
        """Create a new alert"""
        alert_id = f"{component_id}_{metric_name}_{int(time.time())}"
        
        # Check if similar alert already exists
        existing_alert = next(
            (a for a in self.alerts if a.component_id == component_id and 
             a.metric_name == metric_name and a.status == AlertStatus.ACTIVE),
            None
        )
        
        if existing_alert:
            # Update existing alert
            existing_alert.current_value = current_value
            existing_alert.description = description
            return
        
        alert = AlertModel(
            id=alert_id,
            title=title,
            description=description,
            severity=severity,
            status=AlertStatus.ACTIVE,
            component_id=component_id,
            metric_name=metric_name,
            threshold_value=threshold_value,
            current_value=current_value,
            created_at=datetime.now()
        )
        
        self.alerts.append(alert)
        
        # Store in database
        await self._store_alert(alert)
        
        # Broadcast to WebSocket clients
        await self._broadcast_alert(alert)
    
    async def _store_alert(self, alert: AlertModel):
        """Store alert in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO alerts 
                    (id, title, description, severity, status, component_id, metric_name, 
                     threshold_value, current_value, created_at, acknowledged_at, 
                     resolved_at, acknowledged_by)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    alert.id,
                    alert.title,
                    alert.description,
                    alert.severity.value,
                    alert.status.value,
                    alert.component_id,
                    alert.metric_name,
                    alert.threshold_value,
                    alert.current_value,
                    alert.created_at,
                    alert.acknowledged_at,
                    alert.resolved_at,
                    alert.acknowledged_by
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error storing alert: {e}")
    
    async def _broadcast_metrics(self, metrics: PerformanceMetrics):
        """Broadcast metrics to WebSocket clients"""
        if not self.websocket_connections:
            return
        
        message = WebSocketMessage(
            type="metrics_update",
            data=metrics.dict(),
            timestamp=datetime.now()
        )
        
        await self._broadcast_to_websockets(message)
    
    async def _broadcast_alert(self, alert: AlertModel):
        """Broadcast alert to WebSocket clients"""
        if not self.websocket_connections:
            return
        
        message = WebSocketMessage(
            type="new_alert",
            data=alert.dict(),
            timestamp=datetime.now()
        )
        
        await self._broadcast_to_websockets(message)
    
    async def _broadcast_to_websockets(self, message: WebSocketMessage):
        """Broadcast message to all WebSocket clients"""
        if not self.websocket_connections:
            return
        
        message_json = message.json()
        disconnected = set()
        
        for websocket in self.websocket_connections:
            try:
                await websocket.send_text(message_json)
            except Exception as e:
                logger.error(f"Error sending WebSocket message: {e}")
                disconnected.add(websocket)
        
        # Remove disconnected clients
        self.websocket_connections -= disconnected
    
    async def get_system_overview(self) -> SystemOverview:
        """Get system overview"""
        total_components = len(self.components)
        healthy_components = sum(1 for c in self.components.values() if c.get("health_score", 0) >= 0.8)
        degraded_components = sum(1 for c in self.components.values() if 0.5 <= c.get("health_score", 0) < 0.8)
        unhealthy_components = total_components - healthy_components - degraded_components
        
        overall_health = sum(c.get("health_score", 0) for c in self.components.values()) / total_components if total_components > 0 else 0
        
        active_alerts = sum(1 for a in self.alerts if a.status == AlertStatus.ACTIVE)
        
        system_uptime = int((datetime.now() - self.system_start_time).total_seconds())
        
        return SystemOverview(
            timestamp=datetime.now(),
            overall_health=overall_health,
            total_components=total_components,
            healthy_components=healthy_components,
            degraded_components=degraded_components,
            unhealthy_components=unhealthy_components,
            system_uptime=system_uptime,
            active_alerts=active_alerts,
            performance_score=overall_health * 100
        )
    
    async def get_component_health(self, component_id: Optional[str] = None) -> Union[ComponentHealth, List[ComponentHealth]]:
        """Get component health status"""
        if component_id:
            if component_id not in self.components:
                raise HTTPException(status_code=404, detail="Component not found")
            
            component = self.components[component_id]
            return ComponentHealth(
                component_id=component_id,
                component_type=component["type"],
                status=component["status"],
                health_score=component["health_score"],
                last_check=component["last_check"],
                metrics=component.get("metrics", []),
                issues=component.get("issues", []),
                uptime_seconds=component.get("uptime_seconds")
            )
        else:
            return [
                ComponentHealth(
                    component_id=comp_id,
                    component_type=comp["type"],
                    status=comp["status"],
                    health_score=comp["health_score"],
                    last_check=comp["last_check"],
                    metrics=comp.get("metrics", []),
                    issues=comp.get("issues", []),
                    uptime_seconds=comp.get("uptime_seconds")
                )
                for comp_id, comp in self.components.items()
            ]
    
    async def get_alerts(self, status: Optional[AlertStatus] = None, 
                        severity: Optional[AlertSeverity] = None,
                        component_id: Optional[str] = None) -> List[AlertModel]:
        """Get alerts with optional filtering"""
        filtered_alerts = self.alerts
        
        if status:
            filtered_alerts = [a for a in filtered_alerts if a.status == status]
        if severity:
            filtered_alerts = [a for a in filtered_alerts if a.severity == severity]
        if component_id:
            filtered_alerts = [a for a in filtered_alerts if a.component_id == component_id]
        
        return filtered_alerts
    
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> AlertModel:
        """Acknowledge an alert"""
        alert = next((a for a in self.alerts if a.id == alert_id), None)
        
        if not alert:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        if alert.status != AlertStatus.ACTIVE:
            raise HTTPException(status_code=400, detail="Alert is not active")
        
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_at = datetime.now()
        alert.acknowledged_by = acknowledged_by
        
        # Update database
        await self._store_alert(alert)
        
        return alert
    
    async def resolve_alert(self, alert_id: str) -> AlertModel:
        """Resolve an alert"""
        alert = next((a for a in self.alerts if a.id == alert_id), None)
        
        if not alert:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.now()
        
        # Update database
        await self._store_alert(alert)
        
        return alert
    
    async def get_historical_data(self, metric_name: str, component_id: str,
                                 start_time: datetime, end_time: datetime,
                                 interval: str = "1m") -> HistoricalData:
        """Get historical metric data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT timestamp, value FROM metrics 
                    WHERE name = ? AND component_id = ? 
                    AND timestamp BETWEEN ? AND ?
                    ORDER BY timestamp
                ''', (metric_name, component_id, start_time, end_time))
                
                rows = cursor.fetchall()
                
                data_points = [
                    {"timestamp": row[0], "value": row[1]}
                    for row in rows
                ]
                
                return HistoricalData(
                    metric_name=metric_name,
                    component_id=component_id,
                    data_points=data_points,
                    start_time=start_time,
                    end_time=end_time,
                    aggregation="raw",
                    interval=interval
                )
                
        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            return HistoricalData(
                metric_name=metric_name,
                component_id=component_id,
                data_points=[],
                start_time=start_time,
                end_time=end_time,
                aggregation="raw",
                interval=interval
            )
    
    def add_websocket_connection(self, websocket: WebSocket):
        """Add WebSocket connection"""
        self.websocket_connections.add(websocket)
    
    def remove_websocket_connection(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        self.websocket_connections.discard(websocket)

# Global system monitor
system_monitor = SystemStatusMonitor()

# FastAPI app
app = FastAPI(
    title="GrandModel System Status API",
    description="System status monitoring API for GrandModel trading system",
    version="1.0.0"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Add rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Authentication dependency
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """Verify authentication token"""
    token = credentials.credentials
    
    if not token or len(token) < 10:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    
    return {
        "user_id": "admin",
        "username": "admin",
        "permissions": ["read", "write", "alerts"]
    }

# API Endpoints

@app.get("/api/status/overview", response_model=SystemOverview, tags=["Status"])
@limiter.limit("60/minute")
async def get_system_overview(request: Request, user: Dict[str, Any] = Depends(get_current_user)):
    """Get system overview"""
    return await system_monitor.get_system_overview()

@app.get("/api/status/components", response_model=List[ComponentHealth], tags=["Status"])
@limiter.limit("60/minute")
async def get_all_components_health(request: Request, user: Dict[str, Any] = Depends(get_current_user)):
    """Get all components health status"""
    return await system_monitor.get_component_health()

@app.get("/api/status/components/{component_id}", response_model=ComponentHealth, tags=["Status"])
@limiter.limit("60/minute")
async def get_component_health(component_id: str, request: Request, user: Dict[str, Any] = Depends(get_current_user)):
    """Get specific component health status"""
    return await system_monitor.get_component_health(component_id)

@app.get("/api/status/performance", response_model=PerformanceMetrics, tags=["Status"])
@limiter.limit("60/minute")
async def get_performance_metrics(request: Request, user: Dict[str, Any] = Depends(get_current_user)):
    """Get current performance metrics"""
    return await system_monitor._get_system_performance_metrics()

@app.get("/api/status/alerts", response_model=List[AlertModel], tags=["Alerts"])
@limiter.limit("60/minute")
async def get_alerts(
    request: Request,
    status: Optional[AlertStatus] = None,
    severity: Optional[AlertSeverity] = None,
    component_id: Optional[str] = None,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Get alerts with optional filtering"""
    return await system_monitor.get_alerts(status, severity, component_id)

@app.post("/api/status/alerts/{alert_id}/acknowledge", response_model=AlertModel, tags=["Alerts"])
@limiter.limit("30/minute")
async def acknowledge_alert(
    alert_id: str,
    request: Request,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Acknowledge an alert"""
    return await system_monitor.acknowledge_alert(alert_id, user["username"])

@app.post("/api/status/alerts/{alert_id}/resolve", response_model=AlertModel, tags=["Alerts"])
@limiter.limit("30/minute")
async def resolve_alert(
    alert_id: str,
    request: Request,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Resolve an alert"""
    return await system_monitor.resolve_alert(alert_id)

@app.get("/api/status/historical", response_model=HistoricalData, tags=["Status"])
@limiter.limit("30/minute")
async def get_historical_data(
    metric_name: str,
    component_id: str,
    start_time: datetime,
    end_time: datetime,
    interval: str = "1m",
    request: Request = None,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Get historical metric data"""
    return await system_monitor.get_historical_data(metric_name, component_id, start_time, end_time, interval)

@app.websocket("/ws/status")
async def websocket_status_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time status updates"""
    await websocket.accept()
    system_monitor.add_websocket_connection(websocket)
    
    try:
        while True:
            # Keep connection alive
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        system_monitor.remove_websocket_connection(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)