"""
System Monitor for Operations System

This module provides comprehensive system monitoring capabilities including
resource monitoring, performance tracking, and health checks.
"""

import logging


import asyncio
import psutil
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import structlog
import json
from collections import deque

from ..core.event_bus import EventBus, Event, EventType

logger = structlog.get_logger()


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics"""
    GAUGE = "gauge"
    COUNTER = "counter"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    memory_total: float
    memory_available: float
    disk_usage: float
    disk_total: float
    disk_available: float
    network_io_sent: float
    network_io_recv: float
    load_average: List[float]
    process_count: int
    open_files: int
    connections: int
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "memory_total": self.memory_total,
            "memory_available": self.memory_available,
            "disk_usage": self.disk_usage,
            "disk_total": self.disk_total,
            "disk_available": self.disk_available,
            "network_io_sent": self.network_io_sent,
            "network_io_recv": self.network_io_recv,
            "load_average": self.load_average,
            "process_count": self.process_count,
            "open_files": self.open_files,
            "connections": self.connections,
            "custom_metrics": self.custom_metrics
        }


@dataclass
class AlertRule:
    """Alert rule definition"""
    rule_id: str
    name: str
    description: str
    metric_name: str
    condition: str  # e.g., "cpu_usage > 80"
    threshold: float
    severity: AlertSeverity
    duration: int = 60  # seconds
    enabled: bool = True
    cooldown: int = 300  # seconds
    last_triggered: Optional[datetime] = None
    
    def should_trigger(self, value: float, current_time: datetime) -> bool:
        """Check if alert should trigger"""
        if not self.enabled:
            return False
        
        # Check cooldown
        if self.last_triggered:
            if (current_time - self.last_triggered).total_seconds() < self.cooldown:
                return False
        
        # Evaluate condition
        return self._evaluate_condition(value)
    
    def _evaluate_condition(self, value: float) -> bool:
        """Evaluate alert condition"""
        if ">" in self.condition:
            return value > self.threshold
        elif "<" in self.condition:
            return value < self.threshold
        elif ">=" in self.condition:
            return value >= self.threshold
        elif "<=" in self.condition:
            return value <= self.threshold
        elif "==" in self.condition:
            return abs(value - self.threshold) < 0.001
        else:
            return False


@dataclass
class HealthCheck:
    """Health check definition"""
    check_id: str
    name: str
    description: str
    check_function: Callable
    interval: int = 60  # seconds
    timeout: int = 30
    enabled: bool = True
    last_run: Optional[datetime] = None
    last_result: Optional[Dict[str, Any]] = None
    failure_count: int = 0
    max_failures: int = 3


class SystemMonitor:
    """Comprehensive system monitoring"""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.is_running = False
        self.monitoring_task = None
        
        # Metrics storage
        self.metrics_history = deque(maxlen=1000)
        self.current_metrics: Optional[SystemMetrics] = None
        
        # Alert rules
        self.alert_rules: Dict[str, AlertRule] = {}
        
        # Health checks
        self.health_checks: Dict[str, HealthCheck] = {}
        
        # Custom metrics
        self.custom_metrics: Dict[str, Any] = {}
        
        # Monitoring configuration
        self.monitoring_interval = 10  # seconds
        self.retention_hours = 24
        
        # Statistics
        self.alerts_triggered = 0
        self.health_checks_run = 0
        self.health_checks_failed = 0
        
        # Initialize default alert rules
        self._initialize_default_alert_rules()
        
        # Initialize default health checks
        self._initialize_default_health_checks()
        
        logger.info("System Monitor initialized")
    
    def _initialize_default_alert_rules(self):
        """Initialize default alert rules"""
        default_rules = [
            AlertRule(
                rule_id="cpu_high",
                name="High CPU Usage",
                description="CPU usage exceeds 80%",
                metric_name="cpu_usage",
                condition="cpu_usage > 80",
                threshold=80.0,
                severity=AlertSeverity.WARNING
            ),
            AlertRule(
                rule_id="cpu_critical",
                name="Critical CPU Usage",
                description="CPU usage exceeds 95%",
                metric_name="cpu_usage",
                condition="cpu_usage > 95",
                threshold=95.0,
                severity=AlertSeverity.CRITICAL
            ),
            AlertRule(
                rule_id="memory_high",
                name="High Memory Usage",
                description="Memory usage exceeds 85%",
                metric_name="memory_usage",
                condition="memory_usage > 85",
                threshold=85.0,
                severity=AlertSeverity.WARNING
            ),
            AlertRule(
                rule_id="memory_critical",
                name="Critical Memory Usage",
                description="Memory usage exceeds 95%",
                metric_name="memory_usage",
                condition="memory_usage > 95",
                threshold=95.0,
                severity=AlertSeverity.CRITICAL
            ),
            AlertRule(
                rule_id="disk_high",
                name="High Disk Usage",
                description="Disk usage exceeds 85%",
                metric_name="disk_usage",
                condition="disk_usage > 85",
                threshold=85.0,
                severity=AlertSeverity.WARNING
            ),
            AlertRule(
                rule_id="disk_critical",
                name="Critical Disk Usage",
                description="Disk usage exceeds 95%",
                metric_name="disk_usage",
                condition="disk_usage > 95",
                threshold=95.0,
                severity=AlertSeverity.CRITICAL
            )
        ]
        
        for rule in default_rules:
            self.alert_rules[rule.rule_id] = rule
    
    def _initialize_default_health_checks(self):
        """Initialize default health checks"""
        self.health_checks["system_health"] = HealthCheck(
            check_id="system_health",
            name="System Health Check",
            description="Overall system health assessment",
            check_function=self._check_system_health,
            interval=60
        )
        
        self.health_checks["process_health"] = HealthCheck(
            check_id="process_health",
            name="Process Health Check",
            description="Check for zombie processes and high resource usage",
            check_function=self._check_process_health,
            interval=120
        )
        
        self.health_checks["network_health"] = HealthCheck(
            check_id="network_health",
            name="Network Health Check",
            description="Check network connectivity and performance",
            check_function=self._check_network_health,
            interval=60
        )
    
    async def start_monitoring(self):
        """Start system monitoring"""
        if self.is_running:
            logger.warning("System monitoring already running")
            return
        
        self.is_running = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("System monitoring started")
    
    async def stop_monitoring(self):
        """Stop system monitoring"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("System monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_running:
            try:
                # Collect metrics
                metrics = await self._collect_metrics()
                self.current_metrics = metrics
                self.metrics_history.append(metrics)
                
                # Check alert rules
                await self._check_alert_rules(metrics)
                
                # Run health checks
                await self._run_health_checks()
                
                # Cleanup old data
                await self._cleanup_old_data()
                
                # Publish metrics event
                await self._publish_metrics_event(metrics)
                
            except Exception as e:
                logger.error("Error in monitoring loop", error=str(e))
            
            await asyncio.sleep(self.monitoring_interval)
    
    async def _collect_metrics(self) -> SystemMetrics:
        """Collect system metrics"""
        try:
            # CPU metrics
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            memory_total = memory.total / (1024**3)  # GB
            memory_available = memory.available / (1024**3)  # GB
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_usage = (disk.used / disk.total) * 100
            disk_total = disk.total / (1024**3)  # GB
            disk_available = disk.free / (1024**3)  # GB
            
            # Network metrics
            network = psutil.net_io_counters()
            network_io_sent = network.bytes_sent / (1024**2)  # MB
            network_io_recv = network.bytes_recv / (1024**2)  # MB
            
            # Load average
            load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0.0, 0.0, 0.0]
            
            # Process metrics
            process_count = len(psutil.pids())
            
            # File descriptors (approximate)
            open_files = 0
            try:
                open_files = len(psutil.Process().open_files())
            except (FileNotFoundError, IOError, OSError) as e:
                logger.error(f'Error occurred: {e}')
            
            # Network connections
            connections = len(psutil.net_connections())
            
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                memory_total=memory_total,
                memory_available=memory_available,
                disk_usage=disk_usage,
                disk_total=disk_total,
                disk_available=disk_available,
                network_io_sent=network_io_sent,
                network_io_recv=network_io_recv,
                load_average=list(load_avg),
                process_count=process_count,
                open_files=open_files,
                connections=connections,
                custom_metrics=self.custom_metrics.copy()
            )
            
        except Exception as e:
            logger.error("Error collecting metrics", error=str(e))
            # Return default metrics
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_usage=0.0,
                memory_usage=0.0,
                memory_total=0.0,
                memory_available=0.0,
                disk_usage=0.0,
                disk_total=0.0,
                disk_available=0.0,
                network_io_sent=0.0,
                network_io_recv=0.0,
                load_average=[0.0, 0.0, 0.0],
                process_count=0,
                open_files=0,
                connections=0
            )
    
    async def _check_alert_rules(self, metrics: SystemMetrics):
        """Check alert rules against current metrics"""
        current_time = datetime.now()
        
        for rule in self.alert_rules.values():
            if not rule.enabled:
                continue
            
            # Get metric value
            metric_value = getattr(metrics, rule.metric_name, 0.0)
            
            # Check if alert should trigger
            if rule.should_trigger(metric_value, current_time):
                await self._trigger_alert(rule, metric_value, metrics)
    
    async def _trigger_alert(self, rule: AlertRule, value: float, metrics: SystemMetrics):
        """Trigger an alert"""
        rule.last_triggered = datetime.now()
        self.alerts_triggered += 1
        
        # Create alert event
        alert_event = Event(
            type=EventType.ALERT,
            payload={
                "rule_id": rule.rule_id,
                "rule_name": rule.name,
                "severity": rule.severity.value,
                "metric_name": rule.metric_name,
                "metric_value": value,
                "threshold": rule.threshold,
                "message": f"{rule.name}: {rule.metric_name} = {value:.2f}",
                "timestamp": datetime.now().isoformat(),
                "system_metrics": metrics.to_dict()
            }
        )
        
        await self.event_bus.publish(alert_event)
        
        logger.warning(
            "Alert triggered",
            rule_id=rule.rule_id,
            rule_name=rule.name,
            severity=rule.severity.value,
            metric_value=value,
            threshold=rule.threshold
        )
    
    async def _run_health_checks(self):
        """Run health checks"""
        current_time = datetime.now()
        
        for check in self.health_checks.values():
            if not check.enabled:
                continue
            
            # Check if it's time to run
            if check.last_run:
                if (current_time - check.last_run).total_seconds() < check.interval:
                    continue
            
            # Run health check
            await self._run_health_check(check)
    
    async def _run_health_check(self, check: HealthCheck):
        """Run a single health check"""
        check.last_run = datetime.now()
        self.health_checks_run += 1
        
        try:
            # Run check with timeout
            result = await asyncio.wait_for(
                check.check_function(),
                timeout=check.timeout
            )
            
            check.last_result = result
            check.failure_count = 0
            
            # Publish health check result
            health_event = Event(
                type=EventType.HEALTH_CHECK,
                payload={
                    "check_id": check.check_id,
                    "check_name": check.name,
                    "status": "passed",
                    "result": result,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            await self.event_bus.publish(health_event)
            
        except Exception as e:
            check.failure_count += 1
            self.health_checks_failed += 1
            
            check.last_result = {
                "status": "failed",
                "error": str(e),
                "failure_count": check.failure_count
            }
            
            # Publish health check failure
            health_event = Event(
                type=EventType.HEALTH_CHECK,
                payload={
                    "check_id": check.check_id,
                    "check_name": check.name,
                    "status": "failed",
                    "error": str(e),
                    "failure_count": check.failure_count,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            await self.event_bus.publish(health_event)
            
            logger.error(
                "Health check failed",
                check_id=check.check_id,
                check_name=check.name,
                error=str(e),
                failure_count=check.failure_count
            )
    
    async def _check_system_health(self) -> Dict[str, Any]:
        """System health check"""
        try:
            # Check system uptime
            uptime = time.time() - psutil.boot_time()
            
            # Check system load
            load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0.0, 0.0, 0.0]
            
            # Check available resources
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            health_score = 100.0
            issues = []
            
            # Deduct points for high resource usage
            if memory.percent > 90:
                health_score -= 20
                issues.append("High memory usage")
            
            if (disk.used / disk.total) * 100 > 90:
                health_score -= 20
                issues.append("High disk usage")
            
            if load_avg[0] > psutil.cpu_count() * 2:
                health_score -= 15
                issues.append("High system load")
            
            return {
                "health_score": health_score,
                "uptime_hours": uptime / 3600,
                "load_average": list(load_avg),
                "memory_usage": memory.percent,
                "disk_usage": (disk.used / disk.total) * 100,
                "issues": issues,
                "status": "healthy" if health_score > 80 else "degraded" if health_score > 60 else "unhealthy"
            }
            
        except Exception as e:
            return {
                "health_score": 0.0,
                "status": "error",
                "error": str(e)
            }
    
    async def _check_process_health(self) -> Dict[str, Any]:
        """Process health check"""
        try:
            processes = list(psutil.process_iter(['pid', 'name', 'status', 'cpu_percent', 'memory_percent']))
            
            zombie_processes = [p for p in processes if p.info['status'] == psutil.STATUS_ZOMBIE]
            high_cpu_processes = [p for p in processes if p.info['cpu_percent'] > 80]
            high_memory_processes = [p for p in processes if p.info['memory_percent'] > 20]
            
            return {
                "total_processes": len(processes),
                "zombie_processes": len(zombie_processes),
                "high_cpu_processes": len(high_cpu_processes),
                "high_memory_processes": len(high_memory_processes),
                "status": "healthy" if len(zombie_processes) == 0 else "degraded"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def _check_network_health(self) -> Dict[str, Any]:
        """Network health check"""
        try:
            # Check network interfaces
            interfaces = psutil.net_if_stats()
            active_interfaces = [name for name, stats in interfaces.items() if stats.isup]
            
            # Check network connections
            connections = psutil.net_connections()
            established_connections = [c for c in connections if c.status == 'ESTABLISHED']
            
            return {
                "active_interfaces": len(active_interfaces),
                "total_connections": len(connections),
                "established_connections": len(established_connections),
                "status": "healthy" if len(active_interfaces) > 0 else "degraded"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def _cleanup_old_data(self):
        """Clean up old monitoring data"""
        cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)
        
        # Clean up metrics history
        while (self.metrics_history and 
               self.metrics_history[0].timestamp < cutoff_time):
            self.metrics_history.popleft()
    
    async def _publish_metrics_event(self, metrics: SystemMetrics):
        """Publish metrics event"""
        metrics_event = Event(
            type=EventType.METRICS,
            payload={
                "metrics": metrics.to_dict(),
                "timestamp": datetime.now().isoformat()
            }
        )
        
        await self.event_bus.publish(metrics_event)
    
    def add_alert_rule(self, rule: AlertRule) -> bool:
        """Add an alert rule"""
        try:
            self.alert_rules[rule.rule_id] = rule
            logger.info("Alert rule added", rule_id=rule.rule_id, rule_name=rule.name)
            return True
        except Exception as e:
            logger.error("Failed to add alert rule", rule_id=rule.rule_id, error=str(e))
            return False
    
    def remove_alert_rule(self, rule_id: str) -> bool:
        """Remove an alert rule"""
        if rule_id in self.alert_rules:
            del self.alert_rules[rule_id]
            logger.info("Alert rule removed", rule_id=rule_id)
            return True
        return False
    
    def add_health_check(self, check: HealthCheck) -> bool:
        """Add a health check"""
        try:
            self.health_checks[check.check_id] = check
            logger.info("Health check added", check_id=check.check_id, check_name=check.name)
            return True
        except Exception as e:
            logger.error("Failed to add health check", check_id=check.check_id, error=str(e))
            return False
    
    def remove_health_check(self, check_id: str) -> bool:
        """Remove a health check"""
        if check_id in self.health_checks:
            del self.health_checks[check_id]
            logger.info("Health check removed", check_id=check_id)
            return True
        return False
    
    def set_custom_metric(self, name: str, value: Any):
        """Set a custom metric"""
        self.custom_metrics[name] = value
    
    def get_current_metrics(self) -> Optional[SystemMetrics]:
        """Get current system metrics"""
        return self.current_metrics
    
    def get_metrics_history(self, hours: int = 1) -> List[SystemMetrics]:
        """Get metrics history"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [m for m in self.metrics_history if m.timestamp >= cutoff_time]
    
    def get_monitor_status(self) -> Dict[str, Any]:
        """Get monitor status"""
        return {
            "is_running": self.is_running,
            "monitoring_interval": self.monitoring_interval,
            "retention_hours": self.retention_hours,
            "alert_rules_count": len(self.alert_rules),
            "health_checks_count": len(self.health_checks),
            "metrics_history_count": len(self.metrics_history),
            "alerts_triggered": self.alerts_triggered,
            "health_checks_run": self.health_checks_run,
            "health_checks_failed": self.health_checks_failed,
            "custom_metrics_count": len(self.custom_metrics)
        }