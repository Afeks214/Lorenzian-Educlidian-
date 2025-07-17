"""
Comprehensive RTO (Recovery Time Objective) monitoring system for database and trading engine components.

This module provides real-time monitoring of RTO targets:
- Database: <30s recovery target
- Trading Engine: <5s recovery target

Features:
- Real-time RTO measurement and tracking
- Historical trend analysis
- Automated alerting on RTO breaches
- Comprehensive validation testing
- Performance dashboards
"""

import asyncio
import time
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum
from dataclasses import dataclass, field, asdict
from pathlib import Path
import sqlite3
import threading
from contextlib import asynccontextmanager, contextmanager
import statistics
import psutil
import httpx
from concurrent.futures import ThreadPoolExecutor

from src.utils.redis_compat import redis_client
from src.core.event_bus import EventBus
from src.monitoring.health_monitor import HealthStatus, ComponentHealth

logger = logging.getLogger(__name__)

class RTOTarget(Enum):
    """RTO target definitions."""
    DATABASE = 30  # 30 seconds
    TRADING_ENGINE = 5  # 5 seconds

class RTOStatus(Enum):
    """RTO status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    BREACH = "breach"
    CRITICAL = "critical"

@dataclass
class RTOEvent:
    """RTO event data structure."""
    component: str
    event_type: str  # 'failure', 'recovery_start', 'recovery_complete'
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "component": self.component,
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }

@dataclass
class RTOMetric:
    """RTO measurement data."""
    component: str
    target_seconds: float
    actual_seconds: float
    status: RTOStatus
    timestamp: datetime
    failure_start: Optional[datetime] = None
    recovery_start: Optional[datetime] = None
    recovery_complete: Optional[datetime] = None
    details: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def breach_percentage(self) -> float:
        """Calculate breach percentage."""
        if self.actual_seconds <= self.target_seconds:
            return 0.0
        return ((self.actual_seconds - self.target_seconds) / self.target_seconds) * 100
    
    @property
    def is_breach(self) -> bool:
        """Check if RTO is breached."""
        return self.actual_seconds > self.target_seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "component": self.component,
            "target_seconds": self.target_seconds,
            "actual_seconds": self.actual_seconds,
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "failure_start": self.failure_start.isoformat() if self.failure_start else None,
            "recovery_start": self.recovery_start.isoformat() if self.recovery_start else None,
            "recovery_complete": self.recovery_complete.isoformat() if self.recovery_complete else None,
            "breach_percentage": self.breach_percentage,
            "is_breach": self.is_breach,
            "details": self.details
        }

class RTODatabase:
    """SQLite database for RTO metrics storage."""
    
    def __init__(self, db_path: str = "rto_metrics.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS rto_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    component TEXT NOT NULL,
                    target_seconds REAL NOT NULL,
                    actual_seconds REAL NOT NULL,
                    status TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    failure_start TEXT,
                    recovery_start TEXT,
                    recovery_complete TEXT,
                    breach_percentage REAL,
                    is_breach BOOLEAN,
                    details TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS rto_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    component TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_component_timestamp 
                ON rto_metrics(component, timestamp)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_component_timestamp 
                ON rto_events(component, timestamp)
            """)
    
    def store_metric(self, metric: RTOMetric):
        """Store RTO metric."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO rto_metrics (
                    component, target_seconds, actual_seconds, status, timestamp,
                    failure_start, recovery_start, recovery_complete,
                    breach_percentage, is_breach, details
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metric.component,
                metric.target_seconds,
                metric.actual_seconds,
                metric.status.value,
                metric.timestamp.isoformat(),
                metric.failure_start.isoformat() if metric.failure_start else None,
                metric.recovery_start.isoformat() if metric.recovery_start else None,
                metric.recovery_complete.isoformat() if metric.recovery_complete else None,
                metric.breach_percentage,
                metric.is_breach,
                json.dumps(metric.details)
            ))
    
    def store_event(self, event: RTOEvent):
        """Store RTO event."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO rto_events (component, event_type, timestamp, metadata)
                VALUES (?, ?, ?, ?)
            """, (
                event.component,
                event.event_type,
                event.timestamp.isoformat(),
                json.dumps(event.metadata)
            ))
    
    def get_recent_metrics(self, component: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent metrics for a component."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM rto_metrics 
                WHERE component = ? AND timestamp >= ?
                ORDER BY timestamp DESC
            """, (component, cutoff.isoformat()))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_breach_count(self, component: str, hours: int = 24) -> int:
        """Get breach count for a component."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT COUNT(*) FROM rto_metrics 
                WHERE component = ? AND timestamp >= ? AND is_breach = 1
            """, (component, cutoff.isoformat()))
            
            return cursor.fetchone()[0]
    
    def get_average_rto(self, component: str, hours: int = 24) -> float:
        """Get average RTO for a component."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT AVG(actual_seconds) FROM rto_metrics 
                WHERE component = ? AND timestamp >= ?
            """, (component, cutoff.isoformat()))
            
            result = cursor.fetchone()[0]
            return result if result is not None else 0.0

class DatabaseRTOMonitor:
    """Database-specific RTO monitoring."""
    
    def __init__(self, db_config: Dict[str, Any]):
        self.db_config = db_config
        self.target_rto = RTOTarget.DATABASE.value
        self.component_name = "database"
        self._active_failures: Dict[str, datetime] = {}
        self._lock = threading.Lock()
    
    async def check_database_health(self) -> Tuple[bool, float, Dict[str, Any]]:
        """Check database health and measure response time."""
        start_time = time.time()
        details = {}
        
        try:
            # Test database connection
            import psycopg2
            conn = psycopg2.connect(
                host=self.db_config.get('host', 'localhost'),
                port=self.db_config.get('port', 5432),
                database=self.db_config.get('database', 'trading'),
                user=self.db_config.get('user', 'postgres'),
                password=self.db_config.get('password', 'password'),
                connect_timeout=5
            )
            
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            
            response_time = time.time() - start_time
            details['response_time'] = response_time
            details['connection_success'] = True
            
            cursor.close()
            conn.close()
            
            return True, response_time, details
            
        except Exception as e:
            response_time = time.time() - start_time
            details['error'] = str(e)
            details['response_time'] = response_time
            details['connection_success'] = False
            
            return False, response_time, details
    
    async def simulate_recovery(self, failure_id: str) -> float:
        """Simulate database recovery process."""
        logger.info(f"Starting database recovery simulation for {failure_id}")
        
        # Simulate recovery steps
        recovery_steps = [
            ("connection_restore", 2.0),
            ("data_validation", 3.0),
            ("index_rebuild", 8.0),
            ("cache_warmup", 4.0),
            ("health_check", 1.0)
        ]
        
        total_time = 0.0
        for step, duration in recovery_steps:
            await asyncio.sleep(duration)
            total_time += duration
            logger.info(f"Recovery step '{step}' completed in {duration}s")
        
        logger.info(f"Database recovery completed in {total_time}s")
        return total_time
    
    async def trigger_failure_scenario(self, scenario: str = "connection_loss") -> str:
        """Trigger a database failure scenario for testing."""
        failure_id = f"db_failure_{int(time.time())}"
        failure_time = datetime.utcnow()
        
        with self._lock:
            self._active_failures[failure_id] = failure_time
        
        logger.warning(f"Database failure scenario '{scenario}' triggered: {failure_id}")
        
        # Simulate failure duration based on scenario
        failure_durations = {
            "connection_loss": 5.0,
            "disk_full": 15.0,
            "network_partition": 8.0,
            "primary_failure": 25.0
        }
        
        await asyncio.sleep(failure_durations.get(scenario, 10.0))
        return failure_id
    
    def get_failure_start_time(self, failure_id: str) -> Optional[datetime]:
        """Get failure start time."""
        with self._lock:
            return self._active_failures.get(failure_id)
    
    def clear_failure(self, failure_id: str):
        """Clear failure record."""
        with self._lock:
            self._active_failures.pop(failure_id, None)

class TradingEngineRTOMonitor:
    """Trading engine-specific RTO monitoring."""
    
    def __init__(self, engine_config: Dict[str, Any]):
        self.engine_config = engine_config
        self.target_rto = RTOTarget.TRADING_ENGINE.value
        self.component_name = "trading_engine"
        self._active_failures: Dict[str, datetime] = {}
        self._lock = threading.Lock()
    
    async def check_trading_engine_health(self) -> Tuple[bool, float, Dict[str, Any]]:
        """Check trading engine health and measure response time."""
        start_time = time.time()
        details = {}
        
        try:
            # Test trading engine API
            endpoint = self.engine_config.get('health_endpoint', 'http://localhost:8000/health')
            
            async with httpx.AsyncClient() as client:
                response = await client.get(endpoint, timeout=3.0)
                
            response_time = time.time() - start_time
            details['response_time'] = response_time
            details['status_code'] = response.status_code
            details['endpoint'] = endpoint
            
            is_healthy = response.status_code == 200
            return is_healthy, response_time, details
            
        except Exception as e:
            response_time = time.time() - start_time
            details['error'] = str(e)
            details['response_time'] = response_time
            details['endpoint'] = self.engine_config.get('health_endpoint', 'unknown')
            
            return False, response_time, details
    
    async def simulate_recovery(self, failure_id: str) -> float:
        """Simulate trading engine recovery process."""
        logger.info(f"Starting trading engine recovery simulation for {failure_id}")
        
        # Simulate recovery steps (faster than database)
        recovery_steps = [
            ("service_restart", 1.0),
            ("market_data_reconnect", 0.5),
            ("position_reconciliation", 1.5),
            ("risk_system_check", 0.8),
            ("trading_resume", 0.3)
        ]
        
        total_time = 0.0
        for step, duration in recovery_steps:
            await asyncio.sleep(duration)
            total_time += duration
            logger.info(f"Recovery step '{step}' completed in {duration}s")
        
        logger.info(f"Trading engine recovery completed in {total_time}s")
        return total_time
    
    async def trigger_failure_scenario(self, scenario: str = "service_crash") -> str:
        """Trigger a trading engine failure scenario for testing."""
        failure_id = f"engine_failure_{int(time.time())}"
        failure_time = datetime.utcnow()
        
        with self._lock:
            self._active_failures[failure_id] = failure_time
        
        logger.warning(f"Trading engine failure scenario '{scenario}' triggered: {failure_id}")
        
        # Simulate failure duration based on scenario
        failure_durations = {
            "service_crash": 2.0,
            "memory_leak": 3.0,
            "network_timeout": 1.5,
            "config_error": 2.5
        }
        
        await asyncio.sleep(failure_durations.get(scenario, 2.0))
        return failure_id
    
    def get_failure_start_time(self, failure_id: str) -> Optional[datetime]:
        """Get failure start time."""
        with self._lock:
            return self._active_failures.get(failure_id)
    
    def clear_failure(self, failure_id: str):
        """Clear failure record."""
        with self._lock:
            self._active_failures.pop(failure_id, None)

class RTOAlertManager:
    """RTO breach alerting system."""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.alert_cooldown = 300  # 5 minutes
        self._last_alerts: Dict[str, datetime] = {}
        self._lock = threading.Lock()
    
    def should_alert(self, component: str, status: RTOStatus) -> bool:
        """Check if alert should be sent based on cooldown."""
        if status not in [RTOStatus.BREACH, RTOStatus.CRITICAL]:
            return False
        
        with self._lock:
            last_alert = self._last_alerts.get(component)
            now = datetime.utcnow()
            
            if not last_alert:
                self._last_alerts[component] = now
                return True
            
            if (now - last_alert).total_seconds() >= self.alert_cooldown:
                self._last_alerts[component] = now
                return True
        
        return False
    
    async def send_alert(self, metric: RTOMetric):
        """Send RTO breach alert."""
        if not self.should_alert(metric.component, metric.status):
            return
        
        alert_data = {
            "alert_type": "rto_breach",
            "component": metric.component,
            "target_rto": metric.target_seconds,
            "actual_rto": metric.actual_seconds,
            "breach_percentage": metric.breach_percentage,
            "status": metric.status.value,
            "timestamp": metric.timestamp.isoformat(),
            "details": metric.details
        }
        
        # Send to event bus
        await self.event_bus.emit("rto_breach_alert", alert_data)
        
        # Log alert
        logger.critical(
            f"RTO BREACH ALERT: {metric.component} - "
            f"Target: {metric.target_seconds}s, Actual: {metric.actual_seconds}s "
            f"({metric.breach_percentage:.1f}% breach)"
        )
    
    async def send_recovery_alert(self, metric: RTOMetric):
        """Send RTO recovery alert."""
        if metric.status != RTOStatus.HEALTHY:
            return
        
        alert_data = {
            "alert_type": "rto_recovery",
            "component": metric.component,
            "recovery_time": metric.actual_seconds,
            "target_rto": metric.target_seconds,
            "timestamp": metric.timestamp.isoformat(),
            "details": metric.details
        }
        
        # Send to event bus
        await self.event_bus.emit("rto_recovery_alert", alert_data)
        
        logger.info(
            f"RTO RECOVERY: {metric.component} - "
            f"Recovered in {metric.actual_seconds}s (Target: {metric.target_seconds}s)"
        )

class RTOMonitoringSystem:
    """Comprehensive RTO monitoring system."""
    
    def __init__(self, 
                 db_config: Dict[str, Any],
                 engine_config: Dict[str, Any],
                 event_bus: Optional[EventBus] = None):
        self.db_config = db_config
        self.engine_config = engine_config
        self.event_bus = event_bus or EventBus()
        
        # Initialize components
        self.database = RTODatabase()
        self.db_monitor = DatabaseRTOMonitor(db_config)
        self.engine_monitor = TradingEngineRTOMonitor(engine_config)
        self.alert_manager = RTOAlertManager(self.event_bus)
        
        # Monitoring state
        self._monitoring_active = False
        self._monitoring_task = None
        self._lock = threading.Lock()
    
    async def start_monitoring(self, interval: float = 10.0):
        """Start RTO monitoring."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop(interval))
        logger.info("RTO monitoring started")
    
    async def stop_monitoring(self):
        """Stop RTO monitoring."""
        self._monitoring_active = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("RTO monitoring stopped")
    
    async def _monitoring_loop(self, interval: float):
        """Main monitoring loop."""
        while self._monitoring_active:
            try:
                # Check database RTO
                await self._check_database_rto()
                
                # Check trading engine RTO
                await self._check_trading_engine_rto()
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in RTO monitoring loop: {e}")
                await asyncio.sleep(interval)
    
    async def _check_database_rto(self):
        """Check database RTO."""
        is_healthy, response_time, details = await self.db_monitor.check_database_health()
        
        if is_healthy:
            status = RTOStatus.HEALTHY
        else:
            # Determine status based on response time
            if response_time <= self.db_monitor.target_rto:
                status = RTOStatus.WARNING
            elif response_time <= self.db_monitor.target_rto * 2:
                status = RTOStatus.BREACH
            else:
                status = RTOStatus.CRITICAL
        
        metric = RTOMetric(
            component=self.db_monitor.component_name,
            target_seconds=self.db_monitor.target_rto,
            actual_seconds=response_time,
            status=status,
            timestamp=datetime.utcnow(),
            details=details
        )
        
        # Store metric
        self.database.store_metric(metric)
        
        # Send alerts if needed
        await self.alert_manager.send_alert(metric)
    
    async def _check_trading_engine_rto(self):
        """Check trading engine RTO."""
        is_healthy, response_time, details = await self.engine_monitor.check_trading_engine_health()
        
        if is_healthy:
            status = RTOStatus.HEALTHY
        else:
            # Determine status based on response time
            if response_time <= self.engine_monitor.target_rto:
                status = RTOStatus.WARNING
            elif response_time <= self.engine_monitor.target_rto * 2:
                status = RTOStatus.BREACH
            else:
                status = RTOStatus.CRITICAL
        
        metric = RTOMetric(
            component=self.engine_monitor.component_name,
            target_seconds=self.engine_monitor.target_rto,
            actual_seconds=response_time,
            status=status,
            timestamp=datetime.utcnow(),
            details=details
        )
        
        # Store metric
        self.database.store_metric(metric)
        
        # Send alerts if needed
        await self.alert_manager.send_alert(metric)
    
    async def simulate_failure_recovery(self, component: str, scenario: str = "default") -> RTOMetric:
        """Simulate failure and recovery scenario."""
        logger.info(f"Simulating failure/recovery for {component} with scenario '{scenario}'")
        
        if component == "database":
            monitor = self.db_monitor
        elif component == "trading_engine":
            monitor = self.engine_monitor
        else:
            raise ValueError(f"Unknown component: {component}")
        
        # Trigger failure
        failure_start = datetime.utcnow()
        failure_id = await monitor.trigger_failure_scenario(scenario)
        
        # Start recovery
        recovery_start = datetime.utcnow()
        recovery_time = await monitor.simulate_recovery(failure_id)
        recovery_complete = datetime.utcnow()
        
        # Calculate total RTO
        total_rto = (recovery_complete - failure_start).total_seconds()
        
        # Determine status
        if total_rto <= monitor.target_rto:
            status = RTOStatus.HEALTHY
        elif total_rto <= monitor.target_rto * 1.5:
            status = RTOStatus.WARNING
        elif total_rto <= monitor.target_rto * 2:
            status = RTOStatus.BREACH
        else:
            status = RTOStatus.CRITICAL
        
        # Create metric
        metric = RTOMetric(
            component=component,
            target_seconds=monitor.target_rto,
            actual_seconds=total_rto,
            status=status,
            timestamp=datetime.utcnow(),
            failure_start=failure_start,
            recovery_start=recovery_start,
            recovery_complete=recovery_complete,
            details={
                "scenario": scenario,
                "failure_id": failure_id,
                "recovery_time": recovery_time,
                "simulated": True
            }
        )
        
        # Store metric and send alerts
        self.database.store_metric(metric)
        await self.alert_manager.send_alert(metric)
        
        # Clean up
        monitor.clear_failure(failure_id)
        
        return metric
    
    def get_rto_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get RTO summary for dashboard."""
        components = ["database", "trading_engine"]
        summary = {}
        
        for component in components:
            metrics = self.database.get_recent_metrics(component, hours)
            breach_count = self.database.get_breach_count(component, hours)
            avg_rto = self.database.get_average_rto(component, hours)
            
            target_rto = RTOTarget.DATABASE.value if component == "database" else RTOTarget.TRADING_ENGINE.value
            
            summary[component] = {
                "target_rto": target_rto,
                "average_rto": avg_rto,
                "breach_count": breach_count,
                "total_measurements": len(metrics),
                "availability_percentage": ((len(metrics) - breach_count) / len(metrics) * 100) if metrics else 100.0,
                "recent_metrics": metrics[-10:] if metrics else []  # Last 10 measurements
            }
        
        return summary
    
    def get_historical_trends(self, component: str, days: int = 7) -> Dict[str, Any]:
        """Get historical RTO trends."""
        metrics = self.database.get_recent_metrics(component, days * 24)
        
        if not metrics:
            return {"error": "No data available"}
        
        # Convert to numerical data
        times = [datetime.fromisoformat(m['timestamp']) for m in metrics]
        rto_values = [m['actual_seconds'] for m in metrics]
        
        # Calculate trends
        trends = {
            "component": component,
            "period_days": days,
            "total_measurements": len(metrics),
            "min_rto": min(rto_values),
            "max_rto": max(rto_values),
            "avg_rto": statistics.mean(rto_values),
            "median_rto": statistics.median(rto_values),
            "std_dev": statistics.stdev(rto_values) if len(rto_values) > 1 else 0,
            "breach_count": sum(1 for m in metrics if m['is_breach']),
            "breach_percentage": sum(1 for m in metrics if m['is_breach']) / len(metrics) * 100
        }
        
        return trends

# Global RTO monitoring instance
rto_monitor = None

def initialize_rto_monitor(db_config: Dict[str, Any], engine_config: Dict[str, Any]) -> RTOMonitoringSystem:
    """Initialize global RTO monitor."""
    global rto_monitor
    rto_monitor = RTOMonitoringSystem(db_config, engine_config)
    return rto_monitor

def get_rto_monitor() -> Optional[RTOMonitoringSystem]:
    """Get global RTO monitor instance."""
    return rto_monitor